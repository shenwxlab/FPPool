FPs = ['MorganFP', 'EstateFP', 'FragmentFP', 'MACCSFP', 'PubChemFP', 'RdkitFP', 'RGroupFP']
FP_length_dict = {'MorganFP':1024, 'EstateFP':79, 'FragmentFP':86, 'MACCSFP':166, 'PubChemFP':881, 'RdkitFP':1024, 'RGroupFP':2048}
FP_args_list = [{'nBits':1024, 'radius':2}, {}, {}, {}, {}, {'nBits':1024, 'minPath':1, 'maxPath':5}, {'nBits':1024}]
FP_args_dict = {FPs[i]:FP_args_list[i] for i in range(len(FPs))}