"""
File containing high-level functions for RRB decision process.
"""

import pandas as pd
import geopandas as gpd
import numpy as np
import math
import json
from sklearn.metrics import pairwise_kernels


def prepare_data(df, feature_labels, class_label, benefit_label):

    from sklearn.preprocessing import QuantileTransformer

    # df = df.groupby('GEOID10')[feature_labels+[class_label]+[benefit_label]].mean()
    index_transformer = pd.Series(range(len(df))).to_numpy()
    
    X = df[feature_labels].to_numpy()
    b = df[benefit_label].to_numpy()

    frac_faulty = 0.55
    ind_faulty = b.argsort()[:int(len(b) * frac_faulty)]

    tag = df[class_label].to_numpy() * 0
    tag[ind_faulty] = 1

    mask = tag == 1

    X = QuantileTransformer(output_distribution='normal').fit_transform(X)

    return X[~mask][::1], tag[~mask][::1], b[~mask][::1], X[mask][::1], tag[mask][::1], b[mask][::1], index_transformer[~mask][::1], index_transformer[mask][::1]


def MMD2u_estimator(K, m, n):
    """
    Compute the MMD^2_u unbiased statistic.
    This is an implementation of an unbiased MMD^2_u estimator:
    Equation (3) in Gretton et al. Journal of Machine Learning Research 13 (2012) 723-773

    Parameters

    K: numpy-array
        the pair-wise kernel matrix
    m: int
        dimension of the first data set
    n: int
        dimension of the second data set

    Return: float
        an unbiased estimate of MMD^2_u
    """
    K_x = K[:m, :m]
    K_y = K[m:, m:]
    K_xy = K[:m, m:]
    return 1.0 / (m * (m - 1.0)) * (K_x.sum() - K_x.diagonal().sum()) + \
            1.0 / (n * (n - 1.0)) * (K_y.sum() - K_y.diagonal().sum()) - \
            2.0 / (m * n) * K_xy.sum()


def MMD(X, Y, kernel_function='rbf', **kwargs):
    """
       This function performs two sample test. The two-sample hypothesis test is concerned with whether distributions $p$
        and $q$ are different on the basis of finite samples drawn from each of them. This ubiquitous problem appears
        in a legion of applications, ranging from data mining to data analysis and inference. This implementation can
        perform the Kolmogorov-Smirnov test (for one-dimensional data only), Kullback-Leibler divergence and MMD.
        The module perform a bootstrap algorithm to estimate the null distribution, and corresponding p-value.

      Parameters

      X: numpy-array
          Data, of size MxD [M is the number of data points, D is the features dimension]

      Y: numpy-array
          Data, of size NxD [N is the number of data points, D is the features dimension]

      model: string
          defines the basis model to perform two sample test ['KS', 'KL', 'MMD']

      kernel_function: string (optional)
          defines the kernel function, only used for the MMD.
          For the list of implemented kernel please consult with https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.kernel_metrics.html#sklearn.metrics.pairwise.kernel_metrics

      **kwargs:
          extra parameters, these are passed to `pairwise_kernels()` as kernel parameters or `KL_divergence_estimator()`
           as the number of k. E.g., if `kernel_two_sample_test(..., kernel_function='rbf', gamma=0.1)`

      Return: tuple of size 3
          float: the test value,
          numpy-array: a null distribution via bootstraps,
          float: estimated p-value
    """

    m = len(X)
    n = len(Y)

    # compute the test statistics according to the input model
    XY = np.vstack([X, Y])
    K = pairwise_kernels(XY, metric=kernel_function, **kwargs)
    MMD2u_hat = MMD2u_estimator(K, m, n)

    return MMD2u_hat

def witness_function(X, Y, kernel_function='rbf', **kwargs):
    """
     This function computes the witness function. For the definition of the witness function see page 729
     in the "A Kernel Two-Sample Test" by Gretton et al. (2012)

    Parameters

    X: numpy-array
        Data, of size MxD [M is the number of data points, D is the features dimension]

    Y: numpy-array
        Data, of size NxD [N is the number of data points, D is the features dimension]

    kernel_function: string (optional)
        defines the kernel function, only used for the MMD.
        For the list of implemented kernel please consult with https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.kernel_metrics.html#sklearn.metrics.pairwise.kernel_metrics

    **kwargs:
        extra parameters, these are passed to `pairwise_kernels()` as kernel parameters.
        E.g., if `kernel_two_sample_test(..., kernel_function='rbf', gamma=0.1)`

    Return: numpy-array
        witness function
    """

    # data and grid size
    m = len(X)
    n = len(Y)

    # compute pairwise kernels
    K_xg = pairwise_kernels(X, Y, metric=kernel_function, **kwargs)
    K_yg = pairwise_kernels(Y, Y, metric=kernel_function, **kwargs)

    return (np.sum(K_yg, axis=0) / n) - (np.sum(K_xg, axis=0) / m)

def compute_fairness_score(X, Y, KDE_bw, gamma, model='IS-KDE', epsilon=1e-5):
    """

    Parameters
    ----------
    X: numpy-array

    Y: numpy-array

    model: str {'IS-KDE', 'EWF'} (optional)

    epsilon: (optional)

    Returns
    -------
    numpy-array, of size N

    """

    from sklearn.neighbors import KernelDensity

    if model == 'IS-KDE':

        kde = KernelDensity(kernel='tophat', bandwidth=KDE_bw).fit(X)
        p = np.exp(kde.score_samples(Y))
        kde = KernelDensity(kernel='tophat', bandwidth=KDE_bw).fit(Y)
        q = np.exp(kde.score_samples(Y))
        fairness_score = (q+epsilon) / (p+epsilon)
        return fairness_score / np.max(fairness_score)

    elif model == 'EWF':
        ewf = witness_function(X, Y, kernel_function='rbf', gamma=gamma)
        return ( ewf - np.min(ewf) ) / (np.max(ewf) - np.min(ewf))

    else:
        NotImplementedError('Fairness score model %s is not implimented'%model)


def utility_assignment(X, bX, Y, bY, indX, KDE_bw, gamma, n_iterations = 100, frac_improve=0.1, frac_degrade=0.1):

    mmd_hat = [MMD(X, Y, kernel_function='rbf', gamma=gamma)]
    utility_hat = [np.mean(bX)]
    fix_indexes = []
    fss = []

    for i in range(n_iterations):

        lenY = len(Y)
        lenX = len(X)

        # compute fairness score
        fs = compute_fairness_score(X, Y, KDE_bw, gamma)
        bYa = (bY - np.min(bY)) / (np.max(bY) - np.min(bY))

        # pick candidates and change their labels
        # ind_improve = bY.argsort()[-int(lenY * frac_improve):][::-1]
        ind_improve = np.random.choice(range(0, lenY), int(lenY * frac_improve), p=(fs+bYa) / sum(fs+bYa) , replace=False)
        ind_improve_transformed = indX[ind_improve]
        YtoX = Y[ind_improve]
        bYtobX = bY[ind_improve]
        Y = np.delete(Y, ind_improve, axis=0)
        bY = np.delete(bY, ind_improve, axis=0)


        # randomly pick candidates that will degrade overtime and change their labels
        ind_degrade = np.random.choice(range(0, lenX), int(lenX * frac_degrade), replace=False)
        XtoY = X[ind_degrade]
        bXtobY = bX[ind_degrade]
        X = np.delete(X, ind_degrade, axis=0)
        bX = np.delete(bX, ind_degrade, axis=0)

        # update arrays (change their labels)
        Y = np.append(Y, XtoY, axis=0)
        bY = np.append(bY, bXtobY, axis=0)
        X = np.append(X, YtoX, axis=0)
        bX = np.append(bX, bYtobX, axis=0)

        mmd_hat += [MMD(X, Y, kernel_function='rbf', gamma=gamma)]
        utility_hat += [np.mean(bX)]
        fix_indexes += [ind_improve_transformed]
        fss += [fs]

    return X, bX, Y, bY, mmd_hat, utility_hat, fix_indexes, fss


class decision_agent(object):
    def __init__(self, seg, config, N = 10):
        seg['qualities'] = seg['LASTRATING']
        seg = seg[seg['qualities'] > 0]  # discard missing values
        seg['condition'] = 0
        seg.loc[seg['qualities'] < config['qual_threshold'], 'condition'] = 1

        self.gamma = config["eval_gamma"]
        self.KDE_bw = config["eval_KDE_bw"]
        self.frac_improve = config["eval_frac_improve"]
        self.N = N

        self.df_impl = seg

    def impl(self):
        # l2_norm_dict = self.l2_norm_dict
        # df_impl = self.df_impl

        selected_segments = []
        X, tag1, bX, Y, tag2, bY, idX, idY = prepare_data(self.df_impl, feature_labels=['poverty_rate', 'median_household_income'],
        class_label='condition', benefit_label='benefit_score')

        ba = utility_assignment(X, bX, Y, bY, n_iterations=1, frac_improve=self.frac_improve,
                frac_degrade=0.15, indX=idY, KDE_bw=self.KDE_bw, gamma=self.gamma)
        ind = ba[6][0]
        self.df_impl['fairness_score'] = pd.Series(ba[7][0])
        
        return self.df_impl.iloc[ind, :]


class merge_agent(object):
    def __init__(self):
        pass
    def impl(self, segments, df_impl, segments_geoid):
        segments['geometry'] = segments_geoid['geometry'].values

        segments.loc[:,'median_household_income'] = pd.Series([0.0]*len(segments))
        segments.loc[:, 'median_property_value'] = pd.Series([0.0]*len(segments))
        segments.loc[:, 'poverty_rate'] = pd.Series([0.0]*len(segments))
        segments.loc[:, 'benefit_score'] = pd.Series([0.0]*len(segments))
        segments.loc[:, 'blk_grp_num'] = pd.Series([0]*len(segments))

        for i in range(len(df_impl)):
            blk_grp = df_impl.iloc[i]
            median_household_income = blk_grp['median_household_income']
            median_property_value = blk_grp['median_property_value']
            poverty_rate = blk_grp['poverty_rate']
            benefit_score = blk_grp['benefit_scores']
            seg_ids = blk_grp['seg_id'].split(',')
            for j_s in seg_ids:
                if j_s == '':
                    continue
                j = int(j_s)
                segments.at[j, 'blk_grp_num'] += 1
                if segments.at[j, 'blk_grp_num'] == 1:
                    segments.at[j, 'median_household_income'] = median_household_income
                    segments.at[j, 'median_property_value'] = median_property_value
                    segments.at[j, 'poverty_rate'] = poverty_rate
                    segments.at[j, 'benefit_score'] = benefit_score
                else:
                    tmp = segments.at[j, 'median_household_income'] * (segments.at[j, 'blk_grp_num'] - 1)
                    segments.at[j, 'median_household_income'] = tmp + median_household_income / segments.at[j, 'blk_grp_num']
                    tmp = segments.at[j, 'median_property_value'] * (segments.at[j, 'blk_grp_num'] - 1)
                    segments.at[j, 'median_property_value'] = tmp + median_property_value / segments.at[j, 'blk_grp_num']
                    tmp = segments.at[j, 'poverty_rate'] * (segments.at[j, 'blk_grp_num'] - 1)
                    segments.at[j, 'poverty_rate'] = tmp + poverty_rate / segments.at[j, 'blk_grp_num']
                    tmp = segments.at[j, 'benefit_score'] * (segments.at[j, 'blk_grp_num'] - 1)
                    segments.at[j, 'benefit_score'] = tmp + benefit_score / segments.at[j, 'blk_grp_num']
        
        segments = segments.loc[segments['blk_grp_num'] != 0]

        return segments


class RRB(object):
    def __init__(self, config):
        self.fairness_pref = config["fairness_pref"]
        self.benefit_pref = config["benefit_pref"]
        self.a_q = config["a_q"]
        self.q = config["q"]
        self.a_f = config["a_f"]
    
    
    def impl(self, segments, df_impl):
        # Normalize preference parameters.
        fairness_pref = self.fairness_pref
        benefit_pref = self.benefit_pref
        fairness_pref = fairness_pref / (fairness_pref + benefit_pref)
        benefit_pref = benefit_pref / (fairness_pref + benefit_pref)

        # (columns to populate later)
        df_impl.loc[:, 'seg_id'] = ''
        df_impl.loc[:, 'qualities'] = ''
        df_impl['avg_quality'] = pd.Series([-1.0]*len(df_impl))
        df_impl.loc[:, 'lengths'] = ''

        df_impl.set_index('GEOID10', inplace=True)
        # For each road segment
        for j in range(len(segments)):
            seg = segments.iloc[j]
            if seg['LASTRATING'] == 0:  # account for missing values
                continue

            geoid = json.loads(seg['geoid'].replace("'", "\""))

            # Count number of points in road segment
            num_points = 0
            for i in geoid:
                num_points += geoid[i]

            # Append seg_id, qualities, lengths attributes for
            #   each geoid that touches seg
            for gid in geoid:
                if (gid == ''):
                    continue
                tmp = df_impl.at[int(gid),'seg_id']
                tmp_c = tmp
                tmp_c += (str(j)+',')
                df_impl.at[int(gid),'seg_id'] = tmp_c
                tmp = df_impl.at[int(gid),'qualities']
                tmp_c = tmp
                tmp_c += (str(seg['LASTRATING'])+',')
                df_impl.at[int(gid),'qualities'] = tmp_c
                tmp = df_impl.at[int(gid),'lengths']
                tmp_c = tmp
                tmp_c+= (str(seg['LENGTH']*geoid[gid] / num_points)+',')
                df_impl.at[int(gid),'lengths'] = tmp_c
        
        # For each block group, convert 'qualities', 'lengths' to list
        # Use these attributes to compute 'avg_quality'
        for i in df_impl.index:
            qualities_str = df_impl.at[i,'qualities']
            if(qualities_str != ''):
                qualities_str = qualities_str[:(len(qualities_str)-1)]
                qualities = list(map(float,qualities_str.split(',')))
            else:
                qualities =[]
            lengths_str = df_impl.at[i,'lengths']
            if (lengths_str != ''):
                lengths_str = lengths_str[:(len(lengths_str)-1)]
                lengths = list(map(float, lengths_str.split(',')))
            else:
                lengths = []
            tot = 0
            for ind in range(min(len(qualities), len(lengths))):
                tot += lengths[ind] * qualities[ind]
            if (tot != 0):
                tot = tot / sum(lengths)
            else:
                tot = -1.0
            if (math.isnan(tot) != True):
                df_impl.at[i,'avg_quality'] = tot
        df_impl = df_impl.loc[df_impl['avg_quality'] >= 0]

        # Normalize "mhi" column to prevent overflow with np.exp()
        df_impl_copy = df_impl.copy()
        df_impl_copy['median_household_income'] -= df_impl_copy['median_household_income'].min()
        df_impl_copy['median_household_income'] /= df_impl_copy['median_household_income'].max()

        # These are used in calculating the fairness value
        self.i_l = 0 #min(df_impl.loc[df_impl['median_household_income'] >= 0]['median_household_income'])
        self.i_h = 1 #max(df_impl['median_household_income'])

        # These are used in calculating the benefit value
        self.pop_l = min(df_impl.loc[df_impl['population'] >= 0]['population'])
        self.pop_m = max(df_impl['population'])

        # Compute scores for decision process
        weighted_l2norm = []
        fairness_scores = []
        benefit_scores = []
        for i in range(len(df_impl_copy)):
            fair = self.calc_fairness(df_impl_copy.iloc[i]['median_household_income'])
            fairness_scores.append(fair)
            benf = self.calc_benefit( df_impl_copy.iloc[i]['population'])
            benefit_scores.append(benf)
            weighted_l2norm.append(self.calc_weighted_l2norm(benf, fair))

        df_impl.insert(df_impl.shape[1], 'fairness_scores', fairness_scores)
        df_impl.insert(df_impl.shape[1], 'benefit_scores', benefit_scores)
        df_impl.insert(df_impl.shape[1], 'weighted_l2norm', weighted_l2norm)

        return df_impl


    def calc_fairness(self, income):
        if (income <= 0):
            return 0
        i_l = self.i_l
        i_h = self.i_h
        a_f = self.a_f
        return 1 / (1+np.exp(a_f*(income - (i_l+i_h)/2)))


    def calc_benefit(self, pop):
        if (pop < 0):
            return 0
        pop_l = self.pop_l
        pop_m = self.pop_m
        return (pop - pop_l) / (pop_m - pop_l)


    def calc_weighted_l2norm(self, benf, fair):
        fairness_pref = self.fairness_pref
        benefit_pref = self.benefit_pref
        return (fairness_pref*fair**2 + benefit_pref*benf**2)**0.5


def rrb_proc(config):
    print("--- RUNNING RESIDENTIAL ROAD DECISION PROCESS + EVALUATION (step 7) ---\n")

    with open("../data/derived/segments_geoid.csv", "rb") as f:
        segments = pd.read_csv(f)
    with open("../data/derived/df_impl_prep_full.csv", "rb") as f:
        df_impl = pd.read_csv(f)
    segments2 = gpd.read_file('../data/AllRoads_Detroit/AllRoads_Detroit.shp')
    rrb = RRB(config)
    df_impl = rrb.impl(segments, df_impl)
    ma = merge_agent()
    segments3 = ma.impl(segments2, df_impl, segments)
    da = decision_agent(segments3, config)
    results = da.impl()
    results.to_csv('../output/rrb_results.csv')