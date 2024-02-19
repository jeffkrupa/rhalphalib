from __future__ import print_function, division
import sys
import os
import rhalphalib as rl
import numpy as np
import scipy.stats
import pickle
import ROOT
import json
import pandas as pd
import argparse
from array import array
rl.util.install_roofit_helpers()
#rl.ParametericSample.PreferRooParametricHist = False

parser = argparse.ArgumentParser(description='Rhalphalib setup.')
parser.add_argument("--opath", action='store', type=str, required=True, help="Path to store output.")
parser.add_argument("--ipt", action='store', type=int, required=True, help="TF pt order.")
parser.add_argument("--irho", action='store', type=int, required=True, help="TF rho order.")
parser.add_argument("--tagger", action='store', type=str, required=True, help="Tagger name to cut, for example pnmd2prong_ddt.")
parser.add_argument("--pickle", action='store', type=str, required=False, help="Path to pickle holding templates.")
parser.add_argument("--root_path", action='store', type=str, required=True, help="Path to ROOT holding templates.")
parser.add_argument("--all_signals", action='store_true', help="Run on all signal templates.")
parser.add_argument("--MCTF", action='store_true', help="Prefit the TF params to MC.")
parser.add_argument("--do_systematics", action='store_true', help="Include systematics.")
parser.add_argument("--is_blinded", action='store_true', help="Run on 10pct dataset.")
parser.add_argument("--year", action='store', type=str, help="Year to run on : one of 2016APV, 2016, 2017, 2018.")
args = parser.parse_args()

tagger = args.tagger
SF = {
    "2017" :  {
        "V_SF" : 0.896,
        "V_SF_err" : 0.065,
        "shift_SF" : 0.99,
        "shift_SF_err" : 0.004, 
    }
}

lumi_dict = {
    "2017" : 41100
}

with open("xsec.json") as f:
    xsec_dict = json.load(f)

short_to_long = {
    "wqq" : "WJetsToQQ",
    "zqq" : "ZJetsToQQ",
    "tt"  : "TTbar",
    "st"  : "SingleTop",
    "wlnu": "WJetsToLNu",
    "m50" : "VectorZPrimeToQQ_M50",
    "m75" : "VectorZPrimeToQQ_M75",
    "m100" : "VectorZPrimeToQQ_M100",
    "m125" : "VectorZPrimeToQQ_M125",
    "m150" : "VectorZPrimeToQQ_M150",
    "m200" : "VectorZPrimeToQQ_M200",
    "m250" : "VectorZPrimeToQQ_M250",
    #"m300" : "VectorZPrimeToQQ_M300",
}

sys_types = {
    'JES' : 'lnN',
    'JER' : 'lnN',
    'UES' : 'lnN',
    'jet_trigger' : 'lnN',
    'btagEffStat' : 'lnN',
    'btagWeight' : 'lnN',
    'pileup_weight' : 'lnN',
    'Z_d2kappa_EW' : 'lnN',
    'Z_d3kappa_EW' : 'lnN',
    'd1kappa_EW' : 'lnN',
    'd1K_NLO' : 'lnN',
    'd2K_NLO' : 'lnN',
    'd3K_NLO' : 'lnN',
    'L1Prefiring' : 'lnN',
    'scalevar_7pt' : 'lnN',
    'scalevar_3pt' : 'lnN',
    'mu_trigger' : 'lnN',
    'mu_isoweight' : 'lnN',
    'mu_idweight' : 'lnN',
    'HEM18' : 'lnN',
}


sample_maps = {
    "QCD" : ["QCD_HT500to700","QCD_HT700to1000","QCD_HT1000to1500","QCD_HT1500to2000","QCD_HT2000toInf"],
    "wqq" : ["WJetsToQQ_HT-600to800","WJetsToQQ_HT-800toInf"],
    "zqq" : ["ZJetsToQQ_HT-600to800","ZJetsToQQ_HT-800toInf"],
    "tt"  : ["TTTo2L2Nu","TTToHadronic","TTToSemiLeptonic"],
    "JetHT_2017" : ["JetHT_Run2017B","JetHT_Run2017C","JetHT_Run2017D","JetHT_Run2017E","JetHT_Run2017F"], 
    "VectorZPrimeToQQ_M50" : ["VectorZPrimeToQQ_M50"],
    "VectorZPrimeToQQ_M75" : ["VectorZPrimeToQQ_M75"],
    "VectorZPrimeToQQ_M100" : ["VectorZPrimeToQQ_M100"],
    "VectorZPrimeToQQ_M125" : ["VectorZPrimeToQQ_M125"],
    "VectorZPrimeToQQ_M150" : ["VectorZPrimeToQQ_M150"],
    "VectorZPrimeToQQ_M200" : ["VectorZPrimeToQQ_M200"],
    "VectorZPrimeToQQ_M250" : ["VectorZPrimeToQQ_M250"],
    #"VectorZPrimeToQQ_M300" : ["VectorZPrimeToQQ_M300"],
}

if args.all_signals:
    signals = ["m50","m75","m100","m125","m150","m200","m250",]
else:
    signals = ["m150"]

poly_order = (args.ipt,args.irho)
def expo_sample(norm, scale, obs):
    cdf = scipy.stats.expon.cdf(scale=scale, x=obs.binning) * norm
    return (np.diff(cdf), obs.binning, obs.name)


def gaus_sample(norm, loc, scale, obs):
    cdf = scipy.stats.norm.cdf(loc=loc, scale=scale, x=obs.binning) * norm
    return (np.diff(cdf), obs.binning, obs.name)

def plot_mctf(tf_MCtempl, msdbins, name):
    """
    Plot the MC pass / fail TF as function of (pt,rho) and (pt,msd)
    """
    import matplotlib.pyplot as plt

    # arrays for plotting pt vs msd                    
    pts = np.linspace(525,1200,10)
    ptpts, msdpts = np.meshgrid(pts[:-1] + 0.5 * np.diff(pts), msdbins[:-1] + 0.5 * np.diff(msdbins), indexing='ij')
    ptpts_scaled = (ptpts - 525.) / (1200. - 525.)
    rhopts = 2*np.log(msdpts/ptpts)

    rhopts_scaled = (rhopts - (-5.5)) / ((-2.) - (-5.5))
    validbins = (rhopts_scaled >= 0) & (rhopts_scaled <= 1)

    ptpts = ptpts[validbins]
    msdpts = msdpts[validbins]
    ptpts_scaled = ptpts_scaled[validbins]
    rhopts_scaled = rhopts_scaled[validbins]

    tf_MCtempl_vals = tf_MCtempl(ptpts_scaled, rhopts_scaled, nominal=True)
    #print(tf_MCtempl_vals)
    df = pd.DataFrame([])
    df['msd'] = msdpts.reshape(-1)
    df['pt'] = ptpts.reshape(-1)
    df['MCTF'] = tf_MCtempl_vals.reshape(-1)
    #print(df['MCTF'])
    fig, ax = plt.subplots()
    h = ax.hist2d(x=df["msd"],y=df["pt"],weights=df["MCTF"], bins=(msdbins,pts))
    plt.xlabel("$m_{sd}$ [GeV]")
    plt.ylabel("$p_{T}$ [GeV]")
    cb = fig.colorbar(h[3],ax=ax)
    cb.set_label("Ratio")
    fig.savefig(f"{opath}/MCTF_msdpt_"+name+".png",)
    fig.savefig(f"{opath}/MCTF_msdpt_"+name+".pdf",)
    plt.clf()

    # arrays for plotting pt vs rho                                          
    rhos = np.linspace(-5.5,-2.,300)
    ptpts, rhopts = np.meshgrid(pts[:-1] + 0.5*np.diff(pts), rhos[:-1] + 0.5 * np.diff(rhos), indexing='ij')
    ptpts_scaled = (ptpts - 525.) / (1500. - 525.)
    rhopts_scaled = (rhopts - (-5.5)) / ((-2.) - (-5.5))
    validbins = (rhopts_scaled >= 0) & (rhopts_scaled <= 1)

    ptpts = ptpts[validbins]
    rhopts = rhopts[validbins]
    ptpts_scaled = ptpts_scaled[validbins]
    rhopts_scaled = rhopts_scaled[validbins]

    tf_MCtempl_vals = tf_MCtempl(ptpts_scaled, rhopts_scaled, nominal=True)

    df = pd.DataFrame([])
    df['rho'] = rhopts.reshape(-1)
    df['pt'] = ptpts.reshape(-1)
    df['MCTF'] = tf_MCtempl_vals.reshape(-1)

    fig, ax = plt.subplots()
    h = ax.hist2d(x=df["rho"],y=df["pt"],weights=df["MCTF"],bins=(rhos,pts))
    plt.xlabel("rho")
    plt.ylabel("$p_{T}$ [GeV]")
    cb = fig.colorbar(h[3],ax=ax)
    cb.set_label("Ratio")
    fig.savefig(f"{opath}/MCTF_rhopt_"+name+".png",)
    fig.savefig(f"{opath}/MCTF_rhopt_"+name+".pdf",)

    return

def get_templ(region,sample,ptbin,tagger,syst=None,muon=False,nowarn=False,year="2017"):

    def get_factor(f,subsample):
        factor = 0.
        tree = f.Get("Runs")
        genEventSumw_buffer = array('d',[0.0]) 
        tree.SetBranchAddress("genEventSumw",genEventSumw_buffer)
        sum_genEventSumw = 0.
        for entry in range(tree.GetEntries()):
            tree.GetEntry(entry)
            sum_genEventSumw += genEventSumw_buffer[0]

        lumi = lumi_dict[year]
        if args.is_blinded:
            lumi /= 10.
        xsec = xsec_dict[subsample]
        if sum_genEventSumw > 0.:
            factor = xsec*lumi/sum_genEventSumw
        else:
            raise RuntimeError(f"Factor for sample {subsample} <= 0")
        return factor 

    master_hist = None
    
    hist_str = f"SR_ptbin{ptbin}_{tagger}_{region}"
    file0 = ROOT.TFile(f"{args.root_path}/{sample_maps[sample][0]}.root", "READ")
    hist0 = file0.Get(hist_str)
    master_hist = hist0.Clone("msd")
    master_hist.Reset()

    for subsample in sample_maps[sample]:
        file = ROOT.TFile(f"{args.root_path}/{subsample}.root")
        hist = file.Get(hist_str)
        if "JetHT" not in subsample:
            factor = get_factor(file,subsample)
            hist.Scale(factor)
        master_hist.Add(hist)  # Add to master, uncertainties are handled automatically
        file.Close()

    master_hist.SetTitle("msd;msd;;")
    file0.Close()
    return master_hist

def test_rhalphabet(tmpdir,sig):
    throwPoisson = False

    jec = rl.NuisanceParameter("CMS_jec", "lnN")
    sys_shape_dict = {}
    sys_shape_dict['JES'] = rl.NuisanceParameter('CMS_scale_j_{}'.format(args.year), sys_types['JES'])
    sys_shape_dict['JER'] = rl.NuisanceParameter('CMS_res_j_{}'.format(args.year), sys_types['JER'])
    #don't have UES for now
    sys_shape_dict['UES'] = rl.NuisanceParameter('CMS_ues_j_{}'.format(args.year), sys_types['UES'])
    sys_shape_dict['jet_trigger'] = rl.NuisanceParameter('CMS_trigger_{}'.format(args.year), sys_types['jet_trigger'])

    #don't have mu for now
    sys_shape_dict['mu_trigger'] = rl.NuisanceParameter('CMS_mu_trigger_{}'.format(args.year), sys_types['mu_trigger'])
    sys_shape_dict['mu_isoweight'] = rl.NuisanceParameter('CMS_mu_isoweight_{}'.format(args.year), sys_types['mu_isoweight'])
    sys_shape_dict['mu_idweight'] = rl.NuisanceParameter('CMS_mu_idweight_{}'.format(args.year), sys_types['mu_idweight'])
    sys_shape_dict['pileup_weight'] = rl.NuisanceParameter('CMS_PU_{}'.format(args.year), sys_types['pileup_weight'])
    #don't have HEM for now
    sys_shape_dict['HEM18'] = rl.NuisanceParameter('CMS_HEM_{}'.format(args.year), sys_types['HEM18'])
    sys_shape_dict['L1Prefiring'] = rl.NuisanceParameter('CMS_L1prefire_{}'.format(args.year), sys_types['L1Prefiring'])
    #sys_shape_dict['scalevar_7pt'] = rl.NuisanceParameter('CMS_th_scale7pt', sys_types['scalevar_7pt'])
    #sys_shape_dict['scalevar_3pt'] = rl.NuisanceParameter('CMS_th_scale3pt', sys_types['scalevar_3pt']) 

    sys_eleveto = rl.NuisanceParameter('CMS_gghcc_e_veto_{}'.format(args.year), 'lnN')
    sys_muveto = rl.NuisanceParameter('CMS_gghcc_m_veto_{}'.format(args.year), 'lnN')
    sys_tauveto = rl.NuisanceParameter('CMS_gghcc_tau_veto_{}'.format(args.year), 'lnN')

    sys_veff = rl.NuisanceParameter('CMS_gghcc_veff_{}'.format(args.year), 'lnN')

    lumi = rl.NuisanceParameter("CMS_lumi", "lnN")
    tqqeffSF = rl.IndependentParameter("tqqeffSF", 1.0, 0, 10)
    tqqnormSF = rl.IndependentParameter("tqqnormSF", 1.0, 0, 10)
    #with open(args.pickle, "rb") as f:
    #    df = pickle.load(f)
    ptbins = np.array([525, 575, 625, 700, 800, 1200])
    npt = len(ptbins) - 1
    msdbins = np.linspace(30,350,65)
    msd = rl.Observable("msd", msdbins)

    # here we derive these all at once with 2D array
    ptpts, msdpts = np.meshgrid(ptbins[:-1] + 0.3*np.diff(ptbins), msdbins[:-1] + 0.5 * np.diff(msdbins), indexing="ij")
    rhopts = 2 * np.log(msdpts / ptpts)
    ptscaled = (ptpts - 525.0) / (1200.0 - 525.0)
    rhoscaled = (rhopts - (-5.5)) / ((-2.) - (-5.5))
    validbins = (rhoscaled >= 0.) & (rhoscaled <= 1.)
    rhoscaled[~validbins] = 1  # we will mask these out later

    # Build qcd MC pass+fail model and fit to polynomial
    qcdmodel = rl.Model("qcdmodel")
    qcdpass, qcdfail = 0.0, 0.0

    #df = pd.read_csv("/eos/project/c/contrast/public/cl/www/zprime/bamboo/7Feb23-2prongarbitration-2/results/histograms.pkl")
    for ptbin in range(npt):
        failCh = rl.Channel("ptbin%d%s" % (ptbin, "fail"))
        passCh = rl.Channel("ptbin%d%s" % (ptbin, "pass"))
        qcdmodel.addChannel(failCh)
        qcdmodel.addChannel(passCh)
        #QCD MC template
        ptnorm = 1
        failTempl = get_templ("fail", "QCD", ptbin, tagger)
        passTempl = get_templ("pass", "QCD", ptbin, tagger)
        #print(failTempl)
        failCh.setObservation(failTempl,read_sumw2=True)
        passCh.setObservation(passTempl,read_sumw2=True)
        #print(failCh.getObservation())
        #print(failCh.getObservation()[0].shape)
        qcdfail += failCh.getObservation()[0].sum()
        qcdpass += passCh.getObservation()[0].sum()
    qcdeff = qcdpass / qcdfail
    degs = tuple([int(s) for s in [args.ipt, args.irho]])
    if args.MCTF:
        _inits = np.ones(tuple(n + 1 for n in degs))

        tf_MCtempl = rl.BernsteinPoly("tf_MCtempl", (1,1) , ["pt", "rho"],init_params=_inits, limits=(0, 10))
        tf_MCtempl_params = qcdeff * tf_MCtempl(ptscaled, rhoscaled)
        for ptbin in range(npt):
            failCh = qcdmodel["ptbin%dfail" % ptbin]
            passCh = qcdmodel["ptbin%dpass" % ptbin]
            failObs = failCh.getObservation()[0]
            qcdparams = np.array([rl.IndependentParameter("qcdparam_ptbin%d_msdbin%d" % (ptbin, i), 0) for i in range(msd.nbins)])
            sigmascale = 10.0
            scaledparams = failObs * (1 + sigmascale / np.maximum(1.0, np.sqrt(failObs))) ** qcdparams
            fail_qcd = rl.ParametericSample("ptbin%dfail_qcd" % ptbin, rl.Sample.BACKGROUND, msd, scaledparams)
            #print(fail_qcd._observable.binning)
            failCh.addSample(fail_qcd)
            pass_qcd = rl.TransferFactorSample("ptbin%dpass_qcd" % ptbin, rl.Sample.BACKGROUND, tf_MCtempl_params[ptbin, :], fail_qcd)
            passCh.addSample(pass_qcd)
            validbins[ptbin][0:2] = False
            failCh.mask = validbins[ptbin]
            passCh.mask = validbins[ptbin]
        qcdfit_ws = ROOT.RooWorkspace("qcdfit_ws")
        simpdf, obs = qcdmodel.renderRoofit(qcdfit_ws)
        qcdfit = simpdf.fitTo(
            obs,
            ROOT.RooFit.Extended(True),
            ROOT.RooFit.SumW2Error(True),
            ROOT.RooFit.Strategy(0),
            ROOT.RooFit.Save(),
            ROOT.RooFit.Minimizer("Minuit2", "migrad"),
            ROOT.RooFit.PrintLevel(-1),
            ROOT.RooFit.Verbose(0),
        )
        qcdfit_ws.add(qcdfit)
        # Set parameters to fitted values  
        allparams = dict(zip(qcdfit.nameArray(), qcdfit.valueArray()))
        pvalues = []
        for i, p in enumerate(tf_MCtempl.parameters.reshape(-1)):
            p.value = allparams[p.name]
            pvalues += [p.value]


        if "pytest" not in sys.modules:
            qcdfit_ws.writeToFile(os.path.join(str(tmpdir), "testModel_qcdfit.root"))
        if qcdfit.status() != 0:
            raise RuntimeError("Could not fit qcd")
    
        plot_mctf(tf_MCtempl,msdbins,"all")
        param_names = [p.name for p in tf_MCtempl.parameters.reshape(-1)]
        decoVector = rl.DecorrelatedNuisanceVector.fromRooFitResult(tf_MCtempl.name + "_deco", qcdfit, param_names)
        tf_MCtempl.parameters = decoVector.correlated_params.reshape(tf_MCtempl.parameters.shape)
        tf_MCtempl_params_final = tf_MCtempl(ptscaled, rhoscaled)

    _inits = np.ones(tuple(n + 1 for n in degs))
    tf_dataResidual = rl.BernsteinPoly("tf_dataResidual", poly_order, ["pt", "rho"], init_params=_inits, limits=(0, 10))
    tf_dataResidual_params = tf_dataResidual(ptscaled, rhoscaled)

    if args.MCTF:
        tf_params = qcdeff * tf_MCtempl_params_final * tf_dataResidual_params
    else:
        tf_params = qcdeff * tf_dataResidual_params

    # build actual fit model now
    model = rl.Model(f"{sig}_model")

    for ptbin in range(npt):
        for region in ["pass", "fail"]:
            ch = rl.Channel("ptbin%d%s" % (ptbin, region))
            model.addChannel(ch)

            isPass = region == "pass"
            ptnorm = 1.0

            #for now, consider only one other template (signal)
            templates = {
                "wqq" : get_templ(region, "wqq", ptbin, tagger),
                "zqq" : get_templ(region, "zqq", ptbin, tagger),
                "tt" : get_templ(region, "tt", ptbin, tagger),
                sig   : get_templ(region, short_to_long[sig], ptbin, tagger),
            }
            for sName in templates.keys():
                # some mock expectations
                templ = templates[sName]
                stype = rl.Sample.SIGNAL if sName == sig else rl.Sample.BACKGROUND
                sample = rl.TemplateSample(ch.name + "_" + sName, stype, templ)

                # sample.setParamEffect(jec, jecup_ratio)
                if args.do_systematics:
                    for sys_name, sys_val in sys_shape_dict.items():
                        up   = df[f"{short_to_long[sName]}_msd_{sys_name}Up_{region}_{ptbin}"].to_numpy()
                        down = df[f"{short_to_long[sName]}_msd_{sys_name}Down_{region}_{ptbin}"].to_numpy()
                        sample.setParamEffect(sys_val, up, down)
    
                sample.setParamEffect(lumi, 1.027)

                ch.addSample(sample)

            if throwPoisson:
                yields = np.random.poisson(yields)
            data_obs = get_templ(region, f"JetHT_2017", ptbin, tagger)
            ch.setObservation(data_obs)

            # drop bins outside rho validity
            validbins[ptbin][0:2] = False
            mask = validbins[ptbin]
            ch.mask = mask

    for ptbin in range(npt):
        failCh = model["ptbin%dfail" % ptbin]
        passCh = model["ptbin%dpass" % ptbin]

        qcdparams = np.array([rl.IndependentParameter("qcdparam_ptbin%d_msdbin%d" % (ptbin, i), 0) for i in range(msd.nbins)])
        initial_qcd = failCh.getObservation().astype(float)  # was integer, and numpy complained about subtracting float from it
        for sample in failCh:
            #print(sample.name, sample.getExpectation(nominal=True))
            initial_qcd -= sample.getExpectation(nominal=True)
        if np.any(initial_qcd < 0.0):
            raise ValueError("initial_qcd negative for some bins..", initial_qcd)
        sigmascale = 10  # to scale the deviation from initial
        scaledparams = initial_qcd * (1 + sigmascale / np.maximum(1.0, np.sqrt(initial_qcd))) ** qcdparams
        fail_qcd = rl.ParametericSample("ptbin%dfail_qcd" % ptbin, rl.Sample.BACKGROUND, msd, scaledparams)
        failCh.addSample(fail_qcd)
        pass_qcd = rl.TransferFactorSample("ptbin%dpass_qcd" % ptbin, rl.Sample.BACKGROUND, tf_params[ptbin, :], fail_qcd)
        passCh.addSample(pass_qcd)

        #To add
        #tqqpass = passCh["tqq"]
        #tqqfail = failCh["tqq"]
        #tqqPF = tqqpass.getExpectation(nominal=True).sum() / tqqfail.getExpectation(nominal=True).sum()
        #tqqpass.setParamEffect(tqqeffSF, 1 * tqqeffSF)
        #tqqfail.setParamEffect(tqqeffSF, (1 - tqqeffSF) * tqqPF + 1)
        #tqqpass.setParamEffect(tqqnormSF, 1 * tqqnormSF)
        #tqqfail.setParamEffect(tqqnormSF, 1 * tqqnormSF)

    #To add
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
    global opath
    startopath = f"{args.opath}/{tagger}/ipt{args.ipt}_irho{args.irho}"
    os.system(f"cp rhalphalib_zprime.py {startopath}/rhalphalib_zprime.py")
    for sig in signals:
        opath = f"{startopath}/{sig}/"
        if os.path.exists(opath):
            print(f"Path {opath} exists. Remove with \nrm -rf {opath}")
            sys.exit()
        else:
            os.makedirs(opath)
        test_rhalphabet(opath,sig)
