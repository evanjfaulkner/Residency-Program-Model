import numpy as np
import pandas as pd
from scipy import stats

def sample_rotation(n):
    """
    Sample whether applicants rotated with the UW program
    (needs to be updated to reflect real numbers)
    
    Parameters
    -----------
    n: number of samples
    
    Returns
    -----------
    n samples from a bernoulli distribution with p = 0.05
    """
    return np.random.binomial(1,0.05,n)

def sample_urm(n):
    """
    Sample whether applicants are underrepresented/minorities
    Taken from https://www.aamc.org/news-insights/medical-school-applicants-and-enrollments-hit-record-highs-underrepresented-minorities-lead-surge
    Black/Hispanic applicants account for ~24% of all applicants
    (Mark and Randy said the number is likely lower, going for 15% for now while we wait for more exact numbers)
    
    Parameters
    -----------
    n: number of samples
    
    Returns
    -----------
    n samples from a bernoulli distribution with p = 0.15
    """
    return np.random.binomial(1,0.15,n)

def sample_preference(n):
    """
    Sample whether applicants indicated that they preferred the program
    (needs to be updated with accurate numbers)
    
    Parameters
    -----------
    n: number of samples
    
    Returns
    -----------
    n samples from a bernoulli distribution with p = 0.1
    """
    return np.random.binomial(1,0.1,n)

def sample_step_1(n):
    """
    Sample whether applicants passed STEP 1
    Taken from https://www.usmle.org/sites/default/files/2022-05/USMLE%20Step%20Examination%20Score%20Interpretation%20Guidelines_5_24_22_0.pdf
    
    Parameters
    -----------
    n: number of samples
    
    Returns
    -----------
    n samples from a bernoulli distribution with p = 0.95
    """
    return np.random.binomial(1,0.95,n)

def sample_step_2(n):
    """
    Sample STEP 2 scores
    Distribution from https://www.usmle.org/sites/default/files/2022-05/USMLE%20Step%20Examination%20Score%20Interpretation%20Guidelines_5_24_22_0.pdf
    
    Parameters
    -----------
    n: number of samples
    
    Returns
    -----------
    n samples from a custom distribution which matches the distribution given by USMLE
    """
    low = 155
    high = 300
    scores = np.arange(low,high,5)
    pctl = np.array([0,0.2,0.4,0.7,1,1.3,1.6,2,3,5,8,12,17,23,31,40,50,60,71,81,89,95,98,98.4,98.7,99,99.3,99.6,99.9,100])
    pr = np.array([pctl[i+1]-pctl[i] for i in range(len(pctl)-1)])
    pr = pr/sum(pr)
    step2_dist = stats.rv_discrete(a=0,b=300,name='step2_dist', values=(scores, pr))
    return step2_dist.rvs(size=n)

def sample_class_rank(n):
    """
    Sample class ranks of applicants
    ~needs to be updated to reflect real applicants~
    (threshold = top quartile)
    
    Parameters
    -----------
    n: number of samples
    
    Returns
    -----------
    n samples from a binomial distribution with 3 trials each with p=0.6
    """
    return np.random.binomial(3,0.6,n)

def sample_school_rank(n):
    """
    Sample medical school rankings of applicants
    (threshold = top 50 ranked schools)
    
    Parameters
    -----------
    n: number of samples
    
    Returns
    -----------
    n samples from a binomial distribution with 5 trials each with p=0.5
    """
    df = pd.read_csv('./data/kaggle_grad_applications/Admission_Predict.csv')
    rec_data = df['University Rating'].to_numpy()
    rec_dist = stats.gaussian_kde(rec_data)
    new_rec_data = rec_dist.resample(n).flatten()
    return new_rec_data

def sample_research(n):
    """
    Sample research rating of applicants
    ~needs to be updated to reflect real applicants~
    (threshold = multiple 1st author publications, distinguish between poster, publication, etc.)
    
    Parameters
    -----------
    n: number of samples
    
    Returns
    -----------
    n samples from a binomial distribution with 5 trials each with p=0.5
    """
    return np.random.binomial(5,0.5,n)

def sample_recommendation(n):
    """
    Sample recommendation letter ratings of applicants
    ~needs to be updated to reflect real applicants~
    (looking for key phrases, threshold = top20-25%)
    
    Parameters
    -----------
    n: number of samples
    
    Returns
    -----------
    n samples from a binomial distribution with 5 trials each with p=0.5
    """
    df = pd.read_csv('./data/kaggle_grad_applications/Admission_Predict.csv')
    rec_data = df['LOR '].to_numpy()
    rec_dist = stats.gaussian_kde(rec_data)
    new_rec_data = rec_dist.resample(n).flatten()
    return new_rec_data

def sample_leadership(n):
    """
    Sample leadership or other exceptional characteristic ratings of applicants
    ~needs to be updated to reflect real applicants~
    
    Parameters
    -----------
    n: number of samples
    
    Returns
    -----------
    n samples from a binomial distribution with 5 trials each with p=0.5
    """
    return np.random.binomial(5,0.5,n)