from sbnd.general import utils

inds = [(0,0,0),(0,0,1),(0,1,0),(0,1,1),(1,0,0),(1,0,1),(1,1,0),(1,1,1)]
sub_inds = [(1,1),(0,0)]

matched_inds = utils.get_inds_from_sub_inds(inds,sub_inds,2)
print('get_inds_from_sub_inds: ',matched_inds == [(0, 0, 0), (1, 1, 1), (1, 1, 0), (0, 0, 1)])

matched_inds = utils.get_sub_inds_from_inds(inds,sub_inds,2)
print('get_sub_inds_from_inds: ',matched_inds == [(1, 1), (0, 0)])




