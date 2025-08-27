import matplotlib.pyplot as plt
import numpy as np
import os

PLOT_COLORS = ['red', 'green', 'blue', 'orange']  # Colors for your plots
K = 4           # Number of Gaussians in the mixture model
NUM_TRIALS = 3  # Number of trials to run (can be adjusted for debugging)
UNLABELED = -1  # Cluster label for unlabeled data points (do not change)


def main(is_semi_supervised, trial_num):
    """Problem 3: EM for Gaussian Mixture Models (unsupervised and semi-supervised)"""
    print('Running {} EM algorithm...'
          .format('semi-supervised' if is_semi_supervised else 'unsupervised'))

    # Load dataset
    train_path = os.path.join(os.path.abspath(''), 'train.csv')
    x_all, z_all = load_gmm_dataset(train_path)

    # Split into labeled and unlabeled examples
    labeled_idxs = (z_all != UNLABELED).squeeze()
    x_tilde = x_all[labeled_idxs, :]   # Labeled examples
    z_tilde = z_all[labeled_idxs, :]   # Corresponding labels
    x = x_all[~labeled_idxs, :]        # Unlabeled examples

    # *** START CODE HERE ***
    # (1) Initialize mu and sigma by splitting the n_examples data points uniformly at random
    # into K groups, then calculating the sample mean and covariance for each group
    n = x.shape[0]
    group = np.random.uniform(low=0, high=K, size=n).astype('int')
    mu = []
    sigma = []
    for i in range(K):
        mu.append(np.mean(x[group==i,:],axis=0))
        sigma.append(np.dot(np.transpose(x - mu[i]), x - mu[i])/n)
    # (2) Initialize phi to place equal probability on each Gaussian
    # phi should be a numpy array of shape (K,)
    phi = np.full(shape=K, fill_value=1/K)
    # (3) Initialize the w values to place equal probability on each Gaussian
    # w should be a numpy array of shape (m, K)
    w = np.full(shape=(n,K), fill_value=1/K)
    # *** END CODE HERE ***

    if is_semi_supervised:
        w = run_semi_supervised_em(x, x_tilde, z_tilde, w, phi, mu, sigma)
    else:
        w = run_em(x, w, phi, mu, sigma)

    # Plot your predictions
    z_pred = np.zeros(n)
    if w is not None:  # Just a placeholder for the starter code
        for i in range(n):
            z_pred[i] = np.argmax(w[i])

    plot_gmm_preds(x, z_pred, is_semi_supervised, plot_id=trial_num)


def run_em(x:np.ndarray, w:np.ndarray, phi:np.ndarray, mu:list, sigma:list):
    """Problem 3(d): EM Algorithm (unsupervised).

    See inline comments for instructions.

    Args:
        x: Design matrix of shape (n_examples, dim).
        w: Initial weight matrix of shape (n_examples, k).
        phi: Initial mixture prior, of shape (k,).
        mu: Initial cluster means, list of k arrays of shape (dim,).
        sigma: Initial cluster covariances, list of k arrays of shape (dim, dim).
        NOTE: This assumption is wrong. The correct form is sigma.shape = (dim, dim). Assuming gaussians to share one covariance matrix.

    Returns:
        Updated weight matrix of shape (n_examples, k) resulting from EM algorithm.
        More specifically, w[i, j] should contain the probability of
        example x^(i) belonging to the j-th Gaussian in the mixture.
    """
    # No need to change any of these parameters
    eps = 1e-3  # Convergence threshold
    max_iter = 1000
    assert isinstance(x, np.ndarray), F"the var x shpould be of type np.ndarray but we got {type(x)} "
    assert isinstance(w, np.ndarray), F"the var w shpould be of type np.ndarray but we got {type(w)} "
    assert isinstance(phi, np.ndarray), F"the var phi shpould be of type np.ndarray but we got {type(phi)} "
    assert isinstance(mu, list), F"the var mu shpould be of type list but we got {type(mu)} "
    assert isinstance(sigma, list), F"the var sigma shpould be of type list but we got {type(sigma)} "
    assert phi.shape[0] == w.shape[1] , F" the phi's row is {phi.shape[0]} and the w's shape is {w.shape[1]}, they should be equal  "

    # Stop when the absolute change in log-likelihood is < eps
    # See below for explanation of the convergence criterion
    number_of_eg = x.shape[0]
    number_of_clusters = len(mu)
    print(F"the type of X is {type(x)}, w:{type(w) }, phi:{type(phi)}, mu:{type(mu)}, sigma:{type(sigma)} and no of eg is {number_of_eg} and no of cluster is {number_of_clusters} ")
    iteration = 0
    ll = prev_ll = None
    w = np.zeros_like(x)
    while iteration < max_iter and (prev_ll is None or np.abs(ll - prev_ll) >= eps):
        # pass  # Just a placeholder for the starter code
        # *** START CODE HERE
        # (1) E-step: Update your estimates in w
        # K = phi.shape[0]
        pxz = P_x_given_z(x, mu, sigma, phi)
        assert isinstance(pxz, np.ndarray)
        w = pxz / np.sum(pxz, axis=1, keepdims=True)
        # # (2) M-step: Update the model parameters phi, mu, and sigma
        # give me the mean of all the clusters in it
        phi = np.mean(w, axis=0)
        #mu and sigma
        wx = np.dot(np.transpose(w), x)
        for c in range(K):
            mu[c] = wx[c] / np.sum(w[:,c])
            sigma[c] = cal_Sigma(x, mu[c], w[:,c])
        # # (3) Compute the log-likelihood of the data to check for convergence.
        # # By log-likelihood, we mean `ll = sum_x[log(sum_z[p(x|z) * p(z)])]`.
        # # We define convergence by the first iteration where abs(ll - prev_ll) < eps.
        # # Hint: For debugging, recall part (a). We showed that ll should be monotonically increasing.
        prev_ll = ll
        pxz = P_x_given_z(x, mu, sigma, phi)
        ll = np.sum(np.log(np.sum(pxz, axis=1)), axis=0)
        if prev_ll is not None and np.abs(ll - prev_ll) < eps: break
        elif iteration % 10 == 0: print('In iteration {}, the ll is {}'.format(iteration, ll))
        iteration += 1
        # *** END CODE HERE ***
    return w


def run_semi_supervised_em(x:np.ndarray, x_tilde:np.ndarray, z_tilde:np.ndarray, w:np.ndarray, phi:np.ndarray, mu:list, sigma:list):
    """Problem 3(e): Semi-Supervised EM Algorithm.

    See inline comments for instructions.

    Args:
        x: Design matrix of unlabeled examples of shape (n_examples_unobs, dim).
        x_tilde: Design matrix of labeled examples of shape (n_examples_obs, dim).
        z_tilde: Array of labels of shape (n_examples_obs, 1).
        w: Initial weight matrix of shape (n_examples, k).
        phi: Initial mixture prior, of shape (k,).
        mu: Initial cluster means, list of k arrays of shape (dim,).
        sigma: Initial cluster covariances, list of k arrays of shape (dim, dim).

    Returns:
        Updated weight matrix of shape (n_examples, k) resulting from semi-supervised EM algorithm.
        More specifically, w[i, j] should contain the probability of
        example x^(i) belonging to the j-th Gaussian in the mixture.
    """
    # No need to change any of these parameters
    alpha = 20.  # Weight for the labeled examples
    eps = 1e-3   # Convergence threshold
    max_iter = 1000

    assert isinstance(x, np.ndarray), F"the var x shpould be of type np.ndarray but we got {type(x)} "
    assert isinstance(w, np.ndarray), F"the var w shpould be of type np.ndarray but we got {type(w)} "
    assert isinstance(phi, np.ndarray), F"the var phi shpould be of type np.ndarray but we got {type(phi)} "
    assert isinstance(mu, list), F"the var mu shpould be of type list but we got {type(mu)} "
    assert isinstance(sigma, list), F"the var sigma shpould be of type list but we got {type(sigma)} "
    assert phi.shape[0] == w.shape[1] , F" the phi's row is {phi.shape[0]} and the w's shape is {w.shape[1]}, they should be equal  "
    # Stop when the absolute change in log-likelihood is < eps
    # See below for explanation of the convergence criterion
    it = 0
    ll = prev_ll = None
    while it < max_iter and (prev_ll is None or np.abs(ll - prev_ll) >= eps):
        # *** START CODE HERE ***
        # (1) E-step: Update your estimates in w
        n_tilde = x_tilde.shape[0]
        K = phi.shape[0]
        pxz = getPxz(x, mu, sigma, phi)
        w = pxz / np.sum(pxz, axis=1, keepdims=True)
        # (2) M-step: Update the model parameters phi, mu, and sigma
        phi = np.mean(w, axis=0)
        wx = np.dot(np.transpose(w), x)
        for c in range(K):
            xtc = x_tilde[(z_tilde==c)[:,0]]
            mu[c] = (wx[c] + alpha * np.sum(xtc)) / (np.sum(w[:, c]) + alpha * n_tilde)
            sigma[c] = cal_sigma_tilde(x, mu[c], w[:,c], xtc, alpha)
        # (3) Compute the log-likelihood of the data to check for convergence.
        # Hint: Make sure to include alpha in your calculation of ll.
        # Hint: For debugging, recall part (a). We showed that ll should be monotonically increasing.
        prev_ll = ll
        pxz = getPxz(x, mu, sigma, phi)
        ptx = getPtx(x_tilde, mu, sigma, z_tilde)
        ll = np.sum(np.log(np.append(np.sum(pxz, axis=1), alpha * ptx)), axis=0)
        if prev_ll is not None and np.abs(ll - prev_ll) < eps: break
        elif it % 10 == 0: print('In iteration {}, the ll is {}'.format(it, ll))
        it += 1
        # *** END CODE HERE ***

    return w

def P_x_given_z(x:np.ndarray, mu:list, sigma:list, phi:np.ndarray)->np.ndarray:
    """ return a array where the row is no of eg in x and column in the no of cluster(taken from len of mu) """
    assert isinstance(x, np.ndarray), F"the var x shpould be of type np.ndarray but we got {type(x)} "
    assert isinstance(phi, np.ndarray), F"the var phi shpould be of type np.ndarray but we got {type(phi)} "
    assert isinstance(mu, list), F"the var mu shpould be of type list but we got {type(mu)} "
    assert isinstance(sigma, list), F"the var sigma shpould be of type list but we got {type(sigma)} "
    n = x.shape[0]
    K = len(mu)
    pxgivenz = np.zeros((n,K))
    for c in range(K):
        for i in range(n):
            pxgivenz[i,c] = np.exp(-0.5 * np.linalg.multi_dot([x[i] - mu[c], np.linalg.inv(sigma[c]), np.transpose(x[i] - mu[c])])) / (np.sqrt(2 * np.pi * np.linalg.det(sigma[c])))
    pxz = pxgivenz * phi
    return pxz
    

# *** START CODE HERE ***
# Helper functions
def getPxz(x, mu, sigma, phi):
    n = x.shape[0]
    K = len(mu)
    pxgivenz = np.zeros((n,K))
    for c in range(K):
        for i in range(n):
            pxgivenz[i,c] = np.exp(-0.5 * np.linalg.multi_dot([x[i] - mu[c], np.linalg.inv(sigma[c]), np.transpose(x[i] - mu[c])])) / (np.sqrt(2 * np.pi * np.linalg.det(sigma[c])))
    pxz = pxgivenz * phi
    return pxz

def getPtx(x_tilde, mu, sigma, z_tilde):
    nt = x_tilde.shape[0]
    ptx = np.zeros((nt))
    for i in range(nt):
        c = int(z_tilde[i][0])
        ptx[i] = np.exp(-0.5 * np.linalg.multi_dot([x_tilde[i] - mu[c], np.linalg.inv(sigma[c]), np.transpose(x_tilde[i] - mu[c])])) / (np.sqrt(2 * np.pi * np.linalg.det(sigma[c])))
    return ptx

def cal_Sigma(x, muc, wc):
    n,dim = x.shape
    sigmac = np.zeros((dim, dim))
    for i in range(n):
        sigmac += wc[i] * np.outer(x[i] - muc, x[i]- muc)
    sigmac = sigmac / np.sum(wc)
    return sigmac

def cal_sigma_tilde(x, muc, wc, xtc, alpha):
    n, dim = x.shape
    ntc = xtc.shape[0]
    sigmac = np.zeros((dim, dim))
    for i in range(n): sigmac += wc[i] * np.outer(x[i] - muc, x[i]- muc)
    for j in range(ntc): sigmac += alpha * np.outer(xtc[j] - muc, xtc[j]- muc)
    sigmac = sigmac / (np.sum(wc) + alpha * ntc)
    return sigmac

# *** END CODE HERE ***


def plot_gmm_preds(x, z, with_supervision, plot_id):
    """Plot GMM predictions on a 2D dataset `x` with labels `z`.

    Write to the output directory, including `plot_id`
    in the name, and appending 'ss' if the GMM had supervision.

    NOTE: You do not need to edit this function.
    """
    plt.figure(figsize=(12, 8))
    plt.title('{} GMM Predictions'.format('Semi-supervised' if with_supervision else 'Unsupervised'))
    plt.xlabel('x_1')
    plt.ylabel('x_2')

    for x_1, x_2, z_ in zip(x[:, 0], x[:, 1], z):
        color = 'gray' if z_ < 0 else PLOT_COLORS[int(z_)]
        alpha = 0.25 if z_ < 0 else 0.75
        plt.scatter(x_1, x_2, marker='.', c=color, alpha=alpha)

    file_name = 'pred{}_{}.pdf'.format('_ss' if with_supervision else '', plot_id)
    save_path = os.path.join('.', file_name)
    plt.savefig(save_path)


def load_gmm_dataset(csv_path):
    """Load dataset for Gaussian Mixture Model.

    Args:
         csv_path: Path to CSV file containing dataset.

    Returns:
        x: NumPy array shape (n_examples, dim)
        z: NumPy array shape (n_exampls, 1)

    NOTE: You do not need to edit this function.
    """

    # Load headers
    with open(csv_path, 'r') as csv_fh:
        headers = csv_fh.readline().strip().split(',')

    # Load features and labels
    x_cols = [i for i in range(len(headers)) if headers[i].startswith('x')]
    z_cols = [i for i in range(len(headers)) if headers[i] == 'z']

    x = np.loadtxt(csv_path, delimiter=',', skiprows=1, usecols=x_cols, dtype=float)
    z = np.loadtxt(csv_path, delimiter=',', skiprows=1, usecols=z_cols, dtype=float)

    if z.ndim == 1:
        z = np.expand_dims(z, axis=-1)

    return x, z


if __name__ == '__main__':
    np.random.seed(22199)
    # Run NUM_TRIALS trials to see how different initializations
    # affect the final predictions with and without supervision
    for t in range(NUM_TRIALS):
        #main(is_semi_supervised=False, trial_num=t)

        # *** START CODE HERE ***
        # Once you've implemented the semi-supervised version,
        # uncomment the following line.
        # You do not need to add any other lines in this code block.
        main(is_semi_supervised=True, trial_num=t)
        # *** END CODE HERE ***
