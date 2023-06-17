import os
import re
import sys
import logging
import pickle
import fnmatch
import operator
import pandas as pd
import numbers

from helpers.utils import get_logger, save_pkl

        
    
def get_corrs(folder, elias, logger):
    corrs, lens, coverage = {}, {}, {}
  
    for fname in os.listdir(folder):
          # check if current path is a file
          if fnmatch.fnmatch(fname, '%s*.pkl' %(elias)):

              name = fname.split('.pkl')[0] 
                            
              if re.search('Rows_', name):
                cleaned_name = name.partition('Rows_')[-1]
              data_f = open(os.path.join(folder, fname) , 'rb')
              df =  pickle.load(data_f) 
              
              if 'anchor' in elias:
                  coverage_len_corr = df['coverage_vol'].corr(df['anchor_LenExpRule'])   
                  cov =  df['coverage_vol'].values
                  rlens = df['anchor_LenExpRule'].values
                  fname = 'anchor_' +   cleaned_name  
                  
              elif 'lore' in elias:
                  coverage_len_corr = df['coverage_vol'].corr(df['lore_LenExpRule'])  
                  cov =  df['coverage_vol'].values
                  rlens = df['lore_LenExpRule'].values
                  fname = 'lore_' +   cleaned_name   
              
              logger.info('now working on {0}'.format(name))
              
              corrs[fname] = coverage_len_corr
              lens[fname] = rlens
              coverage[fname] = cov
    return corrs, lens, coverage
    

in_dir = 'rules'
out_dir = 'artefacts'
logger_name = os.path.join(out_dir, f"correlations.log")
logger = get_logger(logger_name)


corrs_lore, _, _ = get_corrs(in_dir, 'prefinal_loreExp', logger)
corrs_anchor, _, _  = get_corrs(in_dir, 'prefinal_anchorExp', logger)

corrs_lore.update(corrs_anchor)

all_correlations_df = pd.DataFrame(corrs_lore.items(), columns=['fname', 'correlation'])

save_pkl(out_dir, "Correlations_len_coverage_LoreAnchor", 'all_final', all_correlations_df)   
all_correlations_df.to_csv(os.path.join(out_dir,'correlations_len_coverage_loreAnchor.csv'), sep=';')














