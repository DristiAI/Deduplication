from scipy.sparse import hstack
import numpy as np
import pandas as pd
import sklearn
from scipy.cluster.hierarchy import linkage,dendrogram,fcluster
from sklearn.feature_extraction.text import TfidfVectorizer

def cluster_ids(input_data_frame):
    df=input_data_frame
    vectorizer=TfidfVectorizer()
    Xln=vectorizer.fit_transform(df['ln'])
    Xfn=vectorizer.fit_transform(df['fn'])
    Xnm=hstack((Xfn.todense((-1,-1)),Xln.todense((-1,1))))
    merging=linkage(Xnm,method='complete')
    unique_dates=len(np.unique(df['dob']))
    cluster_ids=fcluster(mergings,unique_dates,criterion='distance')
    df_new=pd.DataFrame({'cluster_ids':cluster_ids,'fn':df['fn'].as_matrix(),'ln':df['ln'].as_matrix(),'dob':df['dob']})
    return df_new
l=pd.read_csv('duplicates.csv',usecols=['fn','ln','dob'])
#print(l[0:4].columns)
#print(type(l))
d=cluster_ids(l)
print(d)    
