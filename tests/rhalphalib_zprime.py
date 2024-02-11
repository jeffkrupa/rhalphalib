from __future__ import print_function, division
import sys
import os
import rhalphalib as rl
import numpy as np
import scipy.stats
import pickle
import ROOT
import pandas as pd
import argparse

rl.util.install_roofit_helpers()
rl.ParametericSample.PreferRooParametricHist = False

parser = argparse.ArgumentParser(description='Rhalphalib setup.')
parser.add_argument("--opath", action='store', type=str, required=True, help="Path to store output.")
parser.add_argument("--ipt", action='store', type=int, required=True, help="TF pt order.")
parser.add_argument("--irho", action='store', type=int, required=True, help="TF rho order.")
parser.add_argument("--all_signals", action='store_true', help="Run on all signal templates.")
args = parser.parse_args()

SF = {
    "2017" :  {
        "V_SF" : 0.896,
        "V_SF_err" : 0.065,
        "shift_SF" : 0.99,
        "shift_SF_err" : 0.004, 
    }
}

short_to_long = {
    "wqq" : "WJetsToQQ",
    "zqq" : "ZJetsToQQ",
    "tt"  : "TTbar",
    "st"  : "SingleTop",
    "wlnu": "WJetsToLNu",
    "m50" : "VectorZPrimeToQQ_M50.root",
    "m75" : "VectorZPrimeToQQ_M75.root",
    "m100" : "VectorZPrimeToQQ_M100.root",
    "m125" : "VectorZPrimeToQQ_M125.root",
    "m150" : "VectorZPrimeToQQ_M150.root",
    "m200" : "VectorZPrimeToQQ_M200.root",
    "m250" : "VectorZPrimeToQQ_M250.root",
    "m300" : "VectorZPrimeToQQ_M300.root",
}

if args.all_signals:
    signals = ["m50","m75","m100","m125","m150","m200","m250","m300"]
else:
    signals = ["m150"]

poly_order = (args.ipt,args.irho)
def expo_sample(norm, scale, obs):
    cdf = scipy.stats.expon.cdf(scale=scale, x=obs.binning) * norm
    return (np.diff(cdf), obs.binning, obs.name)


def gaus_sample(norm, loc, scale, obs):
    cdf = scipy.stats.norm.cdf(loc=loc, scale=scale, x=obs.binning) * norm
    return (np.diff(cdf), obs.binning, obs.name)


def test_rhalphabet(tmpdir,sig):
    throwPoisson = False

    jec = rl.NuisanceParameter("CMS_jec", "lnN")
    sys_shape_dict = {}
    #sys_shape_dict["jes"] = rl.NuisanceParameter("CMS_jes", "shape")
    #sys_shape_dict["jer"] = rl.NuisanceParameter("CMS_jer", "shape")
    lumi = rl.NuisanceParameter("CMS_lumi", "lnN")
    tqqeffSF = rl.IndependentParameter("tqqeffSF", 1.0, 0, 10)
    tqqnormSF = rl.IndependentParameter("tqqnormSF", 1.0, 0, 10)

    ptbins = np.array([525, 575, 625, 700, 800, 1500])
    npt = len(ptbins) - 1
    msdbins = np.linspace(40, 400, 40)
    msd = rl.Observable("msd", msdbins)

    # here we derive these all at once with 2D array
    ptpts, msdpts = np.meshgrid(ptbins[:-1] + 0.3 * np.diff(ptbins), msdbins[:-1] + 0.5 * np.diff(msdbins), indexing="ij")
    rhopts = 2 * np.log(msdpts / ptpts)
    ptscaled = (ptpts - 525.0) / (1500.0 - 525.0)
    rhoscaled = (rhopts - (-5.5)) / ((-2.) - (-5.5))
    validbins = (rhoscaled >= 0) & (rhoscaled <= 1)
    rhoscaled[~validbins] = 1  # we will mask these out later

    # Build qcd MC pass+fail model and fit to polynomial
    qcdmodel = rl.Model("qcdmodel")
    qcdpass, qcdfail = 0.0, 0.0


    df = pd.read_csv("/eos/project/c/contrast/public/cl/www/zprime/bamboo/7Feb23-2prongarbitration-2/results/histograms.csv") 
    for ptbin in range(npt):
        failCh = rl.Channel("ptbin%d%s" % (ptbin, "fail"))
        passCh = rl.Channel("ptbin%d%s" % (ptbin, "pass"))
        qcdmodel.addChannel(failCh)
        qcdmodel.addChannel(passCh)
        # mock template
        ptnorm = 1
        #print(ptbin,"pass",df[f"QCD_msd_pass_{ptbin}"].to_numpy(),)
        #print(ptbin,"fail",df[f"QCD_msd_fail_{ptbin}"].to_numpy(),)
        failTempl = (df[f"QCD_msd_fail_{ptbin}"].to_numpy(), msd.binning, "msd") #expo_sample(norm=ptnorm * 1e5, scale=40, obs=msd)
        passTempl = (df[f"QCD_msd_pass_{ptbin}"].to_numpy(), msd.binning, "msd") #expo_sample(norm=ptnorm * 1e3, scale=40, obs=msd)
        #print(
        failCh.setObservation(failTempl)
        passCh.setObservation(passTempl)
        qcdfail += failCh.getObservation().sum()
        qcdpass += passCh.getObservation().sum()

    #print("qcdfail,qcdpass",qcdfail,",",qcdpass)
    #sys.exit()
    qcdeff = qcdpass / qcdfail
    tf_MCtempl = rl.BernsteinPoly("tf_MCtempl", poly_order, ["pt", "rho"], limits=(-10, 10))
    tf_MCtempl_params = qcdeff * tf_MCtempl(ptscaled, rhoscaled)
    for ptbin in range(npt):
        failCh = qcdmodel["ptbin%dfail" % ptbin]
        passCh = qcdmodel["ptbin%dpass" % ptbin]
        failObs = failCh.getObservation()
        qcdparams = np.array([rl.IndependentParameter("qcdparam_ptbin%d_msdbin%d" % (ptbin, i), 0) for i in range(msd.nbins)])
        sigmascale = 10.0
        scaledparams = failObs * (1 + sigmascale / np.maximum(1.0, np.sqrt(failObs))) ** qcdparams
        fail_qcd = rl.ParametericSample("ptbin%dfail_qcd" % ptbin, rl.Sample.BACKGROUND, msd, scaledparams)
        failCh.addSample(fail_qcd)
        pass_qcd = rl.TransferFactorSample("ptbin%dpass_qcd" % ptbin, rl.Sample.BACKGROUND, tf_MCtempl_params[ptbin, :], fail_qcd)
        passCh.addSample(pass_qcd)

        failCh.mask = validbins[ptbin]
        passCh.mask = validbins[ptbin]

    qcdfit_ws = ROOT.RooWorkspace("qcdfit_ws")
    simpdf, obs = qcdmodel.renderRoofit(qcdfit_ws)
    qcdfit = simpdf.fitTo(
        obs,
        ROOT.RooFit.Extended(True),
        ROOT.RooFit.SumW2Error(True),
        ROOT.RooFit.Strategy(2),
        ROOT.RooFit.Save(),
        ROOT.RooFit.Minimizer("Minuit2", "migrad"),
        ROOT.RooFit.PrintLevel(-1),
        ROOT.RooFit.Verbose(0),
    )
    qcdfit_ws.add(qcdfit)
    if "pytest" not in sys.modules:
        qcdfit_ws.writeToFile(os.path.join(str(tmpdir), "testModel_qcdfit.root"))
    if qcdfit.status() != 0:
        raise RuntimeError("Could not fit qcd")
    #sys.exit()
    param_names = [p.name for p in tf_MCtempl.parameters.reshape(-1)]
    decoVector = rl.DecorrelatedNuisanceVector.fromRooFitResult(tf_MCtempl.name + "_deco", qcdfit, param_names)
    tf_MCtempl.parameters = decoVector.correlated_params.reshape(tf_MCtempl.parameters.shape)
    tf_MCtempl_params_final = tf_MCtempl(ptscaled, rhoscaled)
    tf_dataResidual = rl.BernsteinPoly("tf_dataResidual", poly_order, ["pt", "rho"], limits=(-10, 10))
    tf_dataResidual_params = tf_dataResidual(ptscaled, rhoscaled)
    tf_params = qcdeff * tf_MCtempl_params_final * tf_dataResidual_params

    # build actual fit model now
    model = rl.Model(f"{sig}_model")

    for ptbin in range(npt):
        for region in ["pass", "fail"]:
            ch = rl.Channel("ptbin%d%s" % (ptbin, region))
            model.addChannel(ch)

            isPass = region == "pass"
            ptnorm = 1.0
            templates = {
                #"qcd": (df[f"QCD_msd_{region}_{ptbin}"].to_numpy(), msd.binning, "msd"),  
                #"wqq": (df[f"WJetsToQQ_msd_{region}_{ptbin}"].to_numpy(), msd.binning, "msd"),
                #"zqq": (df[f"ZJetsToQQ_msd_{region}_{ptbin}"].to_numpy(), msd.binning, "msd"), 
                #"tt": (df[f"TTbar_msd_{region}_{ptbin}"].to_numpy(), msd.binning, "msd"), 
                #"wlnu": (df[f"WJetsToLNu_msd_{region}_{ptbin}"].to_numpy(), msd.binning, "msd"), 
                #"st": (df[f"SingleTop_msd_{region}_{ptbin}"].to_numpy(), msd.binning, "msd"), 
                #sig : (df[f"{short_to_long[sig]}_msd_{region}_{ptbin}"].to_numpy(), msd.binning, "msd"), 
                #"tqq": #gaus_sample(norm=ptnorm * (40 if isPass else 80), loc=150, scale=20, obs=msd),
            }
            #print(templates["wqq"][0])
            #for sName in ["zqq", "wqq", "m50"]:
            for sName in templates.keys():
                print(short_to_long[sName])
                # some mock expectations
                templ = templates[sName]
                stype = rl.Sample.SIGNAL if sName == sig else rl.Sample.BACKGROUND
                sample = rl.TemplateSample(ch.name + "_" + sName, stype, templ)

                # mock systematics
                #jecup_ratio = np.random.normal(loc=1, scale=0.05, size=msd.nbins)
                #msdUp = np.linspace(0.9, 1.1, msd.nbins)
                #msdDn = np.linspace(1.2, 0.8, msd.nbins)

                # for jec we set lnN prior, shape will automatically be converted to norm systematic
                # sample.setParamEffect(jec, jecup_ratio)
                print("nominal",df[f"{short_to_long[sName]}_msd_{region}_{ptbin}"].to_numpy())
                for sys_name, sys_val in sys_shape_dict.items():
                    up   = df[f"{short_to_long[sName]}_msd_{sys_name}Up_{region}_{ptbin}"].to_numpy()
                    down = df[f"{short_to_long[sName]}_msd_{sys_name}Down_{region}_{ptbin}"].to_numpy()
                    print(sys_name, sys_val, )#
                    print("up",up)
                    print("down",down) 
                    sample.setParamEffect(sys_val, up, down)

                #sample.setParamEffect(lumi, 1.027)

                ch.addSample(sample)

            templates["qcd"] = (df[f"QCD_msd_{region}_{ptbin}"].to_numpy(), msd.binning, "msd")
 
            yields = sum(tpl[0] for itpl, tpl in templates.items() if sig not in itpl)
            if throwPoisson:
                yields = np.random.poisson(yields)
            #data_obs = (yields, msd.binning, msd.name)
            data_obs = (df[f"data_msd_{region}_{ptbin}"].to_numpy(), msd.binning, "msd") #(yields, msd.binning, msd.name)
            ch.setObservation(data_obs)

            # drop bins outside rho validity
            mask = validbins[ptbin]
            # blind bins 11, 12, 13
            # mask[11:14] = False
            ch.mask = mask

    for ptbin in range(npt):
        failCh = model["ptbin%dfail" % ptbin]
        passCh = model["ptbin%dpass" % ptbin]

        qcdparams = np.array([rl.IndependentParameter("qcdparam_ptbin%d_msdbin%d" % (ptbin, i), 0) for i in range(msd.nbins)])
        initial_qcd = failCh.getObservation().astype(float)  # was integer, and numpy complained about subtracting float from it
        for sample in failCh:
            initial_qcd -= sample.getExpectation(nominal=True)
        if np.any(initial_qcd < 0.0):
            raise ValueError("initial_qcd negative for some bins..", initial_qcd)
        sigmascale = 10  # to scale the deviation from initial
        scaledparams = initial_qcd * (1 + sigmascale / np.maximum(1.0, np.sqrt(initial_qcd))) ** qcdparams
        fail_qcd = rl.ParametericSample("ptbin%dfail_qcd" % ptbin, rl.Sample.BACKGROUND, msd, scaledparams)
        failCh.addSample(fail_qcd)
        pass_qcd = rl.TransferFactorSample("ptbin%dpass_qcd" % ptbin, rl.Sample.BACKGROUND, tf_params[ptbin, :], fail_qcd)
        passCh.addSample(pass_qcd)

        #tqqpass = passCh["tqq"]
        #tqqfail = failCh["tqq"]
        #tqqPF = tqqpass.getExpectation(nominal=True).sum() / tqqfail.getExpectation(nominal=True).sum()
        #tqqpass.setParamEffect(tqqeffSF, 1 * tqqeffSF)
        #tqqfail.setParamEffect(tqqeffSF, (1 - tqqeffSF) * tqqPF + 1)
        #tqqpass.setParamEffect(tqqnormSF, 1 * tqqnormSF)
        #tqqfail.setParamEffect(tqqnormSF, 1 * tqqnormSF)

    '''
    # Fill in muon CR
    for region in ["pass", "fail"]:
        ch = rl.Channel("muonCR%s" % (region,))
        model.addChannel(ch)

        isPass = region == "pass"
        templates = {
            "tqq": gaus_sample(norm=10 * (30 if isPass else 60), loc=150, scale=20, obs=msd),
            "qcd": expo_sample(norm=10 * (5e2 if isPass else 1e3), scale=40, obs=msd),
        }
        for sName, templ in templates.items():
            stype = rl.Sample.BACKGROUND
            sample = rl.TemplateSample(ch.name + "_" + sName, stype, templ)

            # mock systematics
            jecup_ratio = np.random.normal(loc=1, scale=0.05, size=msd.nbins)
            sample.setParamEffect(jec, jecup_ratio)

            ch.addSample(sample)

        # make up a data_obs
        templates = {
            "tqq": gaus_sample(norm=10 * (30 if isPass else 60), loc=150, scale=20, obs=msd),
            "qcd": expo_sample(norm=10 * (5e2 if isPass else 1e3), scale=40, obs=msd),
        }
        yields = sum(tpl[0] for tpl in templates.values())
        if throwPoisson:
            yields = np.random.poisson(yields)
        data_obs = (yields, msd.binning, msd.name)
        ch.setObservation(data_obs)

    tqqpass = model["muonCRpass_tqq"]
    tqqfail = model["muonCRfail_tqq"]
    tqqPF = tqqpass.getExpectation(nominal=True).sum() / tqqfail.getExpectation(nominal=True).sum()
    tqqpass.setParamEffect(tqqeffSF, 1 * tqqeffSF)
    tqqfail.setParamEffect(tqqeffSF, (1 - tqqeffSF) * tqqPF + 1)
    tqqpass.setParamEffect(tqqnormSF, 1 * tqqnormSF)
    tqqfail.setParamEffect(tqqnormSF, 1 * tqqnormSF)
    '''
    with open(os.path.join(str(tmpdir), f"{sig}_model.pkl"), "wb") as fout:
        pickle.dump(model, fout)

    model.renderCombine(os.path.join(str(tmpdir), f"{sig}_model"))



if __name__ == "__main__":
    opath = f"{args.opath}_ipt{args.ipt}_irho{args.irho}"
    os.mkdir(opath)
    for sig in signals:
        opath = f"./{opath}/{sig}/"
        if os.path.exists(opath):
            raise RuntimeError(f"Path {opath} exists")
        else:
            os.mkdir(opath)
        test_rhalphabet(opath,sig)
