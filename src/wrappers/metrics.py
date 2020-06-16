from clusim.clustering import Clustering
from sklearn.metrics import adjusted_mutual_info_score


def ami_score(clu1: Clustering, clu2: Clustering):
    """Compute adjusted mutual (AMI) information between two clusterings.

    This method computes the AMI using the scikit-learn method 'adjusted_mutual_info_score',
    because it is significantly faster than it's clusim counterpart.

    Parameters
    ----------
    clu1 : Clustering
        First clustering to compare.
    clu2 : Clustering
        Second clustering to compare.

    Returns
    ------
    float
        Adjusted mutual information between the two input clusterings.
    """
    labels1 = clu1.to_membership_list()
    labels2 = clu2.to_membership_list()
    return adjusted_mutual_info_score(labels1, labels2, average_method='arithmetic')
