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
import uproot
from array import array
from rhalphalib import AffineMorphTemplate, MorphHistW2
rl.util.install_roofit_helpers()
#rl.ParametericSample.PreferRooParametricHist = False
np.random.seed(1)
parser = argparse.ArgumentParser(description='Rhalphalib setup.')
parser.add_argument("--opath", action='store', type=str, required=True, help="Path to store output.")
parser.add_argument("--ipt", action='store', type=int, required=True, help="TF pt order.")
parser.add_argument("--irho", action='store', type=int, required=True, help="TF rho order.")
parser.add_argument("--iptMC", action='store', type=int, required=False, help="MCTF pt order.")
parser.add_argument("--irhoMC", action='store', type=int, required=False, help="MCTF rho order.")
parser.add_argument("--tagger", action='store', type=str, required=True, help="Tagger name to cut, for example pnmd2prong_ddt.")
parser.add_argument("--pickle", action='store', type=str, required=False, help="Path to pickle holding templates.")
parser.add_argument("--sigmass", action='store', type=str, required=False, default="150",help="mass point like 150.")
#parser.add_argument("--root_path", action='store', type=str, required=True, help="Path to ROOT holding templates.")
parser.add_argument("--root_file", action='store', type=str, required=True, help="Path to ROOT holding templates.")
parser.add_argument("--all_signals", action='store_true', help="Run on all signal templates.")
parser.add_argument("--scale_qcd", action='store_true', help="Scale QCD MC so its poisson matches true uncs.")
parser.add_argument("--ftest", action='store_true', default=False, help="Run ftest.")
parser.add_argument("--pseudo", action='store_true', default=False, help="Run pseudo data.")
parser.add_argument("--MCTF", action='store_true', help="Prefit the TF params to MC.")
parser.add_argument("--do_systematics", action='store_true', help="Include systematics.")
parser.add_argument("--is_blinded", action='store_true', help="Run on 10pct dataset.")
parser.add_argument("--throwPoisson", action='store_true', help="Throw poisson.")
parser.add_argument("--year", action='store', type=str, help="Year to run on : one of 2016APV, 2016, 2017, 2018.")
args = parser.parse_args()

tagger = args.tagger
SF = {
    "2017" :  {
        "V_SF" : 0.896,
        "V_SF_ERR" : 0.065,
        "SHIFT_SF" : 0.99,
        "SHIFT_SF_ERR" : 0.004, 
    }
}

lumi_dict = {
    "2017" : 41100
}

lumi_dict_unc = {
    "2016": 1.01,
    "2017": 1.02,
    "2018": 1.015,
}
lumi_correlated_dict_unc = {
    "2016": 1.006,
    "2017": 1.009,
    "2018": 1.02,
}
lumi_1718_dict_unc = {
    "2017": 1.006,
    "2018": 1.002,
}

def smass(sName):
    if 'hbb' in sName:
        _mass = 125.
    elif sName in ['wqq', 'tt', 'st',]:
        _mass = 80.
    elif sName in ['zqq', 'zcc', 'zbb']:
        _mass = 90.
    else:
        raise ValueError("DAFUQ is {}".format(sName))
    return _mass


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
    "dy"  : ["DYJetsToLL_Pt-200To400","DYJetsToLL_Pt-400To600","DYJetsToLL_Pt-600To800","DYJetsToLL_Pt-800To1200","DYJetsToLL_Pt-1200To2500"],
    "st"  : ["ST_tW_antitop_5f_inclusiveDecays_TuneCP5_13TeV-powheg-pythia8","ST_tW_top_5f_inclusiveDecays_TuneCP5_13TeV-powheg-pythia8","ST_t-channel_antitop_4f_InclusiveDecays_TuneCP5_13TeV-powheg-madspin-pythia8","ST_t-channel_top_4f_InclusiveDecays_TuneCP5_13TeV-powheg-madspin-pythia8"],
    "hbb" : ["GluGluHToBB"],
    "wlnu" : ["WJetsToLNu_HT200to400","WJetsToLNu_HT400to600","WJetsToLNu_HT600to800","WJetsToLNu_HT800to1200","WJetsToLNu_HT1200to2500","WJetsToLNu_HT2500toInf"],
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
    signals = ["m"+args.sigmass]

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
    fig.savefig(f"{opath}/MCTF_msdpt_"+name+".png")
    fig.savefig(f"{opath}/MCTF_msdpt_"+name+".pdf")
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

def get_templ(region,sample,ptbin,tagger,syst=None,muon=False,nowarn=False,year="2017",scaledown=False):

    hist_str = f"SR_{sample}_ptbin{ptbin}_{tagger}_{region}"
    if syst is not None:
        hist_str = hist_str +"__"+syst
    #print(hist_str)
    with uproot.open(args.root_file) as f:
        #print(f.keys())
        hist = f[hist_str]
    hist_values = hist.values()
    if scaledown:
        hist_values *= 1e-2

    return (hist_values, hist.axis().edges(), "msd", hist.variances())
       
def th1_to_numpy(path,label="msd"):
    with uproot.open(path) as file:
        th1d = file[label]
        _hist, _ = th1d.to_numpy()
    return _hist

def shape_to_num(region, sName, ptbin, syst_down_up, mask, muon=False, bound=0.5, inflate=False):
    #print(sName)
    _nom = get_templ(region, sName, ptbin, tagger)
    #_nom = th1_to_numpy(path)


    #if template is very small don't add unc
    if _nom[0] is None:
        return None
    _nom_rate = np.sum(_nom[0] * mask)
    if _nom_rate < .1:
        return 1.0
    #ignore one sided for now
    _one_side = None #get_templ(f, region, sName, ptbin, syst=syst, muon=muon, nowarn=True)
    _up = get_templ(region, sName, ptbin, tagger, syst=syst_down_up[1], muon=muon, nowarn=True)

    #_up = th1_to_numpy(path)

    _down = get_templ(region, sName, ptbin, tagger, syst=syst_down_up[0], muon=muon, nowarn=True)
    #_down = th1_to_numpy(path)
    if _up is None and _down is None and _one_side is None:
        return None
    else:
        if _one_side is not None:
            _up_rate = np.sum(_one_side[0] * mask)
            _diff = np.abs(_up_rate - _nom_rate)
            magnitude = _diff / _nom_rate
        elif _down[0] is not None and _up[0] is not None:
            _up_rate = np.sum(_up[0] * mask)
            _down_rate = np.sum(_down[0] * mask)
            #print("_up_rate",_up_rate)
            #print("_down_rate",_down_rate)
            _diff = np.abs(_up_rate - _nom_rate) + np.abs(_down_rate - _nom_rate)
            magnitude = _diff / (2. * _nom_rate)
        else:
            raise NotImplementedError
    if bound is not None:
        magnitude = min(magnitude, bound)
    #inflate uncs while debugging what went wrong
    if inflate: 
        magnitude *= 10 
    #print(magnitude)
    return 1.0 + magnitude

def test_rhalphabet(tmpdir,sig,throwPoisson=False):
    

    jec = rl.NuisanceParameter("CMS_jec", "lnN")
    sys_shape_dict = {}
    sys_shape_dict['JES'] = rl.NuisanceParameter('CMS_scale_j_{}'.format(args.year), sys_types['JES'])
    sys_shape_dict['JER'] = rl.NuisanceParameter('CMS_res_j_{}'.format(args.year), sys_types['JER'])
    #don't have UES for now
    sys_shape_dict['UES'] = rl.NuisanceParameter('CMS_ues_j_{}'.format(args.year), sys_types['UES'])
    sys_shape_dict['jet_trigger'] = rl.NuisanceParameter('CMS_trigger_{}'.format(args.year), sys_types['jet_trigger'])
    sys_shape_dict['L1Prefiring'] = rl.NuisanceParameter('CMS_L1prefire_{}'.format(args.year), sys_types['L1Prefiring'])

    sys_shape_dict['pileup_weight'] = rl.NuisanceParameter('CMS_PU_{}'.format(args.year), sys_types['pileup_weight'])
    #don't have HEM for now
    sys_shape_dict['HEM18'] = rl.NuisanceParameter('CMS_HEM_{}'.format(args.year), sys_types['HEM18'])
    #don't have mu for now
    sys_shape_dict['mu_trigger'] = rl.NuisanceParameter('CMS_mu_trigger_{}'.format(args.year), sys_types['mu_trigger'])
    sys_shape_dict['mu_isoweight'] = rl.NuisanceParameter('CMS_mu_isoweight_{}'.format(args.year), sys_types['mu_isoweight'])
    sys_shape_dict['mu_idweight'] = rl.NuisanceParameter('CMS_mu_idweight_{}'.format(args.year), sys_types['mu_idweight'])
    #sys_shape_dict['scalevar_7pt'] = rl.NuisanceParameter('CMS_th_scale7pt', sys_types['scalevar_7pt'])
    #sys_shape_dict['scalevar_3pt'] = rl.NuisanceParameter('CMS_th_scale3pt', sys_types['scalevar_3pt']) 

    sys_eleveto = rl.NuisanceParameter('CMS_e_veto_{}'.format(args.year), 'lnN')
    sys_muveto = rl.NuisanceParameter('CMS_m_veto_{}'.format(args.year), 'lnN')
    sys_tauveto = rl.NuisanceParameter('CMS_tau_veto_{}'.format(args.year), 'lnN')

    sys_veff = rl.NuisanceParameter('CMS_veff_{}'.format(args.year), 'lnN')

    sys_lumi = rl.NuisanceParameter("CMS_lumi", "lnN")
    sys_lumi_correlated = rl.NuisanceParameter('CMS_lumi_13TeV_correlated', 'lnN')
    sys_lumi_1718 = rl.NuisanceParameter('CMS_lumi_13TeV_1718', 'lnN')


    tqqeffSF = rl.IndependentParameter("tqqeffSF", 1.0, 0, 10)
    tqqnormSF = rl.IndependentParameter("tqqnormSF", 1.0, 0, 10)
    #with open(args.pickle, "rb") as f:
    #    df = pickle.load(f)
    ptbins = np.array([525, 575, 625, 700, 800, 1200])
    npt = len(ptbins) - 1
    msdbins = np.linspace(40,350,63)
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
        if args.MCTF:
            qcdmodel.addChannel(failCh)
            qcdmodel.addChannel(passCh)
    qcdeff = qcdpass / qcdfail
    if args.MCTF:
        degsMC = tuple([int(s) for s in [args.iptMC, args.irhoMC]])
        _initsMC = np.load(f"data/inits_MC_{args.year}.npy")
        #_initsMC = np.ones(tuple(n + 1 for n in degsMC))
        print(_initsMC)
        tf_MCtempl = rl.BasisPoly(f"tf{args.year}_MCtempl",
                                      degsMC, ['pt', 'rho'], basis="Bernstein",
                                      init_params = _initsMC,
                                      limits=(-50, 50), coefficient_transform=None
        )
        #tf_MCtempl = rl.BernsteinPoly("tf_MCtempl", (1,1) , ["pt", "rho"],init_params=_inits, limits=(0, 10))
        tf_MCtempl_params = qcdeff * tf_MCtempl(ptscaled, rhoscaled)
        for ptbin in range(npt):
            failCh = qcdmodel["ptbin%dfail" % ptbin]
            passCh = qcdmodel["ptbin%dpass" % ptbin]
            failObs = failCh.getObservation()[0]
            qcdparams = np.array([rl.IndependentParameter("qcdparam_ptbin%d_msdbin%d" % (ptbin, i), 0) for i in range(msd.nbins)])
            sigmascale = 10.0
            scaledparams = failObs * (1 + sigmascale / np.maximum(1.0, np.sqrt(failObs))) ** qcdparams
            fail_qcd = rl.ParametericSample("ptbin%dfail_qcd" % ptbin, rl.Sample.BACKGROUND, msd, scaledparams)
            print(fail_qcd._observable.binning)
            failCh.addSample(fail_qcd)
            pass_qcd = rl.TransferFactorSample("ptbin%dpass_qcd" % ptbin, rl.Sample.BACKGROUND, tf_MCtempl_params[ptbin, :], fail_qcd)
            passCh.addSample(pass_qcd)
            #validbins[ptbin][0:2] = False
            failCh.mask = validbins[ptbin]
            passCh.mask = validbins[ptbin]
        qcdfit_ws = ROOT.RooWorkspace("qcdfit_ws")
        simpdf, obs = qcdmodel.renderRoofit(qcdfit_ws)
        #gmin = ROOT.Math.Factory.CreateMinimizer("migrad2","minuit")
        #ROOT.Math.MinimizerOptions.SetMaxIterations(gmin,1000000)
        #gmin.Print()
        #minimizer=ROOT.RooFit.Minimizer("Minuit2", "migrad")
        #minimizer.setMaxFunctionCalls(10000000) 
        qcdfit = simpdf.fitTo(
            obs,
            ROOT.RooFit.Extended(True),
            #ROOT.RooFit.AsymptoticError(True),
            ROOT.RooFit.SumW2Error(True),
            ROOT.RooFit.Strategy(2),
            #ROOT.RooFit.MaxCalls(1000000),
            ROOT.RooFit.Save(),
            ROOT.RooFit.Minimizer("Minuit2", "migrad"),
            #ROOT.RooFit.Minimizer("Minuit2", "migrad",ROOT.RooFit.Minimizer.setMaxFunctionCalls(10000000)),
            ROOT.RooFit.PrintLevel(1),
            ROOT.RooFit.Verbose(0),
        )
        '''
        nll = simpdf.createNLL(obs, ROOT.RooFit.Extended(True), ROOT.RooFit.DataError(ROOT.RooAbsData.SumW2))
        minimizer = ROOT.RooMinimizer(nll)
        minimizer.setStrategy(0)  # As you had in your original setup
        minimizer.setMaxFunctionCalls(10000000)  # Set the maximum number of function calls
        minimizer.setMaxIterations(10000000)  # Set the maximum number of iterations
        minimizer.setPrintLevel(1)  # Adjust the verbosity of the output
        minimizer.setErrorLevel(100)  # Adjust the tolerance, higher might help
        minimizer.minimize("Minuit2", "migrad")

        qcdfit = minimizer.save()
        ''' 
        qcdfit_ws.add(qcdfit)
        # Set parameters to fitted values  
        allparams = dict(zip(qcdfit.nameArray(), qcdfit.valueArray()))
        pvalues = []
        for i, p in enumerate(tf_MCtempl.parameters.reshape(-1)):
            p.value = allparams[p.name]
            pvalues += [p.value]

        print(pvalues)
        if "pytest" not in sys.modules:
            qcdfit_ws.writeToFile(os.path.join(str(tmpdir), "testModel_qcdfit.root"))
        if not (qcdfit.status() == 0  or qcdfit.status() == 1):
            raise RuntimeError("Could not fit qcd")
        print("qcdprefit status=",qcdfit.status()) 
        plot_mctf(tf_MCtempl,msdbins,"all")
        param_names = [p.name for p in tf_MCtempl.parameters.reshape(-1)]
        decoVector = rl.DecorrelatedNuisanceVector.fromRooFitResult(tf_MCtempl.name + "_deco", qcdfit, param_names)
        tf_MCtempl.parameters = decoVector.correlated_params.reshape(tf_MCtempl.parameters.shape)
        tf_MCtempl_params_final = tf_MCtempl(ptscaled, rhoscaled)
        #sys.exit()

    degs = tuple([int(s) for s in [args.ipt, args.irho]])
    _inits = np.ones(tuple(n + 1 for n in degs))
    tf_dataResidual = rl.BasisPoly(f"tf{args.year}_dataResidual",
                                       degs, ['pt', 'rho'], basis='Bernstein', init_params=_inits,
                                       limits=(-50, 50), coefficient_transform=None)
    #tf_dataResidual = rl.BernsteinPoly("tf_dataResidual", poly_order, ["pt", "rho"], init_params=_inits, limits=(0, 10))
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

            templates = {
                "wqq"  : get_templ(region, "wqq", ptbin, tagger),
                "zqq"  : get_templ(region, "zqq", ptbin, tagger, scaledown=True if args.pseudo else False),
                "tt"   : get_templ(region, "tt", ptbin, tagger),
                "wlnu" : get_templ(region, "wlnu", ptbin, tagger),
                "dy"   : get_templ(region, "dy", ptbin, tagger),
                "st"   : get_templ(region, "st", ptbin, tagger),
                "hbb"  : get_templ(region, "hbb", ptbin, tagger),  
                "qcd"  : get_templ(region, "QCD", ptbin, tagger),  
            } 

            
            if not args.ftest:
                templates[sig] = get_templ(region, short_to_long[sig], ptbin, tagger)
            mask = validbins[ptbin].copy()

            if args.ftest:
                include_samples = ["zqq"] #qcd here?
            else:
                include_samples = ["wqq","zqq","tt","wlnu","dy","st","hbb",sig]

            for sName in include_samples:
                # some mock expectations
                templ = templates[sName]
                print(ptbin, region, sName) 

                if args.ftest:
                    stype = rl.Sample.SIGNAL if sName == "zqq" else rl.Sample.BACKGROUND
                    #templ[0] = templ[0]*1e-4 #Scale down signal?
                else:
                    stype = rl.Sample.SIGNAL if sName == sig else rl.Sample.BACKGROUND
                sample = rl.TemplateSample(ch.name + "_" + sName, stype, templ)
 

                def smorph(templ, sName):
                    if templ is None:
                        return None
                    if sName not in ['qcd', 'dy', 'wlnu', 'tt', 'st']:
                        return MorphHistW2(templ).get(shift=SF[args.year]['shift_SF'] * smass(sName),
                                                      #smear=SF[year]['smear_SF']
                                                      )
                    else:
                        return templ
                ##https://github.com/nsmith-/rhalphalib/blob/master/rhalphalib/template_morph.py#L45-L58 how do i do this on ROOT templates?
                if args.do_systematics:

                    sample.setParamEffect(sys_lumi, lumi_dict_unc[args.year])
                    sample.setParamEffect(sys_lumi_correlated, lumi_correlated_dict_unc[args.year])
                    if args.year != '2016':
                        sample.setParamEffect(sys_lumi_1718, lumi_1718_dict_unc[args.year])
                    sample.setParamEffect(sys_eleveto, 1.005)
                    sample.setParamEffect(sys_muveto, 1.005)
                    sample.setParamEffect(sys_tauveto, 1.005)

                    sample.autoMCStats(lnN=True)

                    sys_names = [
                        'JES', 'JER', 'jet_trigger','pileup_weight','L1Prefiring',
                        #'Z_d2kappa_EW', 'Z_d3kappa_EW', 'd1kappa_EW', 'd1K_NLO', 'd2K_NLO', 'd3K_NLO',
                        #'scalevar_7pt', 'scalevar_3pt',
                        #'UES','btagEffStat', 'btagWeight',
                    ]
                    sys_name_updown = {
                        'JES' : ["jesTotaldown","jesTotalup"], 'JER' : ["jerdown","jerup"], 'pileup_weight' : ["pudown","puup"], 'jet_trigger' : ["stat_dn","stat_up"], 'L1Prefiring' : ["L1PreFiringup","L1PreFiringdown"],
                    }
                    if stype == rl.Sample.SIGNAL and not args.ftest: 
                        sName = short_to_long[sName]
                    for sys_name in sys_names:
                        if (("NLO" in sys_name) or ("EW" in sys_name)) and not sName in ['zqq', 'wqq']:
                            continue
                        if ("Z_d" in sys_name) and sName not in ['zqq']:
                            continue
                        if sys_shape_dict[sys_name].combinePrior == "lnN":
                            _sys_ef = shape_to_num(region, sName, ptbin,
                                                    sys_name_updown[sys_name], mask, bound=None if 'scalevar' not in sys_name else 0.25,inflate=True)
                            if _sys_ef is None:
                                continue
                            sample.setParamEffect(sys_shape_dict[sys_name], _sys_ef)

                    if sName not in ["qcd", 'dy', 'wlnu','tt','st',]:
                        sample.scale(SF[args.year]['V_SF'])
                        sample.setParamEffect( sys_veff, 1.0 + SF[args.year]['V_SF_ERR'] / SF[args.year]['V_SF'])
                    ###SFs complicated by high-purity bb region...fully insitu using Zbb?

                else:
                    sample.setParamEffect(sys_lumi, lumi_dict_unc[args.year])


                ch.addSample(sample)
                 

            if not args.pseudo:
                data_obs = get_templ(region, f"JetHT_2017", ptbin, tagger)
                if throwPoisson:
                    yields = np.random.poisson(yields)
            else:
                yields = []
 
                for sName in ["QCD"]:
                    _sample = get_templ(region, sName, ptbin, tagger)
                    _sample_yield = _sample[0]
                    if args.scale_qcd: 
                        dummyqcd = rl.TemplateSample("dummyqcd",rl.Sample.BACKGROUND , _sample)
                        nomrate = dummyqcd._nominal
                        downrate = np.sum(np.nan_to_num(dummyqcd._nominal - np.sqrt(dummyqcd._sumw2), 0.0))
                        uprate = np.sum(np.nan_to_num(dummyqcd._nominal + np.sqrt(dummyqcd._sumw2), 0.0))
                        diff = np.sum(np.abs(uprate-nomrate) + np.abs(downrate-nomrate))
                        mean = diff/(2.*np.sum(nomrate))
                        #sqrt(nom*N) = mean -> N = mean**2/nom
                        scale = mean**2/np.sum(nomrate)
                        print("qcdscale needed to match mcstat uncs: using poisson:", 1./scale)
                        _sample_yield = _sample_yield.copy()*1./scale
                    yields.append(_sample_yield)
                yields = np.sum(np.array(yields), axis=0)

                if throwPoisson:
                    yields = np.random.poisson(yields)

                data_obs = (yields, msd.binning, msd.name)

            ch.setObservation(data_obs[0:3])

            # drop bins outside rho validity
            #validbins[ptbin][0:2] = False
            mask = validbins[ptbin]
            ch.mask = mask

    for ptbin in range(npt):
        failCh = model["ptbin%dfail" % ptbin]
        passCh = model["ptbin%dpass" % ptbin]

        qcdparams = np.array([rl.IndependentParameter("qcdparam_ptbin%d_msdbin%d" % (ptbin, i), 0) for i in range(msd.nbins)])
        initial_qcd = failCh.getObservation().astype(float)  # was integer, and numpy complained about subtracting float from it
        print("initial_qcd",initial_qcd)
        for sample in failCh:
            print(sample.name, sample.getExpectation(nominal=True))
            initial_qcd -= sample.getExpectation(nominal=True)
        if args.pseudo:
            initial_qcd[initial_qcd<0] = 0.
        if np.any(initial_qcd < 0.0):
            raise ValueError("initial_qcd negative for some bins..", initial_qcd)
        sigmascale = 10  # to scale the deviation from initial
        scaledparams = initial_qcd * (1 + sigmascale / np.maximum(1.0, np.sqrt(initial_qcd))) ** qcdparams
        fail_qcd = rl.ParametericSample(f"ptbin{ptbin}fail_{args.year}_qcd" , rl.Sample.BACKGROUND, msd, scaledparams)
        failCh.addSample(fail_qcd)
        pass_qcd = rl.TransferFactorSample(f"ptbin{ptbin}pass_{args.year}_qcd" , rl.Sample.BACKGROUND, tf_params[ptbin, :], fail_qcd)
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

    conf_dict = vars(args)
    conf_dict['nbins'] = float(np.sum(validbins))
    print(conf_dict)
    import json
    # Serialize data into file:
    json.dump(conf_dict,
              open("{}/config.json".format(f"{tmpdir}/{sig}_model",), 'w'),
              sort_keys=True,
              indent=4,
              separators=(',', ': '))



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
        test_rhalphabet(opath,sig,args.throwPoisson)
