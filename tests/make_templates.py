import argparse
import ROOT
from array import array
import json

parser = argparse.ArgumentParser(description='Rhalphalib setup.')
parser.add_argument("--root_path", action='store', type=str, required=True, help="Path to ROOT holding templates.")
parser.add_argument("--is_blinded", action='store_true', help="Blinded dataset.")
parser.add_argument("--year", action='store', type=str, help="Year to run on : one of 2016APV, 2016, 2017, 2018.")
args = parser.parse_args()

lumi_dict = {
    "2017" : 41100
}

with open("xsec.json") as f:
    xsec_dict = json.load(f)

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

sys_names = [
    'JES', 'JER', 'jet_trigger','pileup_weight','L1Prefiring',
    #'Z_d2kappa_EW', 'Z_d3kappa_EW', 'd1kappa_EW', 'd1K_NLO', 'd2K_NLO', 'd3K_NLO',
    #'scalevar_7pt', 'scalevar_3pt',
    #'UES','btagEffStat', 'btagWeight',                    
]

sys_name_updown = {
    'JES' : ["jesTotaldown","jesTotalup"], 
    'JER' : ["jerdown","jerup"], 
    'pileup_weight' : ["pudown","puup"], 
    'jet_trigger' : ["stat_dn","stat_up"], 
    'L1Prefiring' : ["L1PreFiringup","L1PreFiringdown"],
}



output_file = ROOT.TFile(args.root_path+f"/TEMPLATES{'_blind' if args.is_blinded else ''}.root", "RECREATE")

def make_templates(region,sample,ptbin,tagger,syst=None,muon=False,nowarn=False,year="2017"):

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
    if syst is not None:
        hist_str += f"__{syst}"
    file0 = ROOT.TFile(f"{args.root_path}/{sample_maps[sample][0]}.root", "READ")
    hist0 = file0.Get(hist_str)
    master_hist = hist0.Clone(hist_str.replace(f"_ptbin{ptbin}",f"_{sample}_ptbin{ptbin}"))#+"_"+sample)
    master_hist.Reset()
    for subsample in sample_maps[sample]:
        file = ROOT.TFile(f"{args.root_path}/{subsample}.root")
        hist = file.Get(hist_str)
        if "JetHT" not in subsample:
            factor = get_factor(file,subsample)
            hist.Scale(factor)
        master_hist.Add(hist)  # Add to master, uncertainties are handled automatically
        file.Close()
    hist_str = hist_str.replace(f"_ptbin{ptbin}",f"_{sample}_ptbin{ptbin}")
    master_hist.SetTitle(hist_str+";"+hist_str+";;")
    output_file.cd()
    master_hist.Write()
    file0.Close()
    return 

for isamp,isamplist in sample_maps.items():
    for tagger in ["pnmd2prong_0p05","pnmd2prong_ddt",]:
        for iptbin in range(0,5):
            for region in ["pass","fail"]:
                make_templates(region,isamp,iptbin,tagger,syst=None,muon=False,nowarn=False,year="2017")
                if "JetHT" in isamp: continue
                for syst in sys_names:
                    make_templates(region,isamp,iptbin,tagger,syst=sys_name_updown[syst][0],muon=False,nowarn=False,year="2017")
                    make_templates(region,isamp,iptbin,tagger,syst=sys_name_updown[syst][1],muon=False,nowarn=False,year="2017")
                #break
output_file.Close()

