
from collections import defaultdict
from sklearn.tree import _tree

def tree_comprehend(tree_id, decision_tree, feature_names=None, precision=3, proportion=False):
    node_map = {}
    node_links = []
    
    def node_to_str(tree, node_id):
        feature = feature_names[tree.feature[node_id]]
        return feature

    def recurse(tree, node_id, criterion, parent=None, depth=0):
        if node_id == _tree.TREE_LEAF:
            raise ValueError("Invalid node_id %s" % _tree.TREE_LEAF)
        left_child = tree.children_left[node_id]
        right_child = tree.children_right[node_id]
        # Collect ranks for 'leaf' option in plot_options
        if left_child == _tree.TREE_LEAF:
            ranks['leaves'].append(node_id)
        elif str(depth) not in ranks:
            ranks[str(depth)] = [node_id]
        else:
            ranks[str(depth)].append(node_id)
        node_map[node_id] = node_to_str(tree, node_id)
        if parent is not None:
            # Add edge to parent
            node_links.append([parent, node_id])
        if left_child != _tree.TREE_LEAF:
            recurse(tree, left_child, criterion=criterion, parent=node_id,
                    depth=depth + 1)
            recurse(tree, right_child, criterion=criterion, parent=node_id,
                    depth=depth + 1)
            
    ranks = {'leaves': []}
    if isinstance(decision_tree, _tree.Tree):
        recurse(decision_tree, 0, criterion="impurity")
    else:
        recurse(decision_tree.tree_, 0, criterion=decision_tree.criterion)
    new_ranks = {}
    for key in ranks.keys():
        new_nodes_at_this_level = []
        nodes_at_this_level = ranks[key]
        if key == 'leaves':
            #key = str(decision_tree.max_depth)
            continue
        for node in nodes_at_this_level:
            new_nodes_at_this_level.append(node_map[node])    
        fq= defaultdict( int )
        for w in new_nodes_at_this_level:
            fq[w] += 1    
        new_ranks[key] = dict(fq)
    df= pd.DataFrame(new_ranks).reset_index().melt(id_vars=['index']).rename(columns={"index":"feature", "variable":"level", "value":"count"}).fillna(0.0)
    df['tree'] = tree_id
    df['present'] = 0
    df.loc[df['count'] > 0, 'present'] = 1
    # Check if a feature is used at all
    df2 = df.copy().drop_duplicates(subset=['feature']).reset_index(drop=True)
    df2['tree'] = tree_id
        
    return df[['tree', 'level', 'feature', 'count', 'present']], df2


def forest_join(spm):
    final = pd.DataFrame()
    final2 = pd.DataFrame()
    for i in range(0, spm.estimator.n_estimators):
        x, y = tree_comprehend(tree_id=i, decision_tree=spm.estimator.estimators_[i], feature_names=spm.predictors)
        final = pd.concat([final, x], axis=0)
        final2 = pd.concat([final2, y], axis=0)
    forest = final.groupby(['level', 'feature']).sum().reset_index()[['level', 'feature', 'count', 'present']]
    return final, forest, final2



# spm is a sklearn RF estimator
df_trees, df_forest, final2 = forest_join(spm)

output_importances_levels = df_forest.pivot(index='feature', columns='level').reset_index().sort_values('feature')

final2['key']=1
final2= final2[['feature', 'tree', 'key']].pivot(index='feature', columns='tree', values='key')
final2['sum_of_trees_with_variable'] = final2.sum(axis=1)
final2 = final2.reset_index()
final2 = final2[['feature', 'sum_of_trees_with_variable']].copy()
final2 = final2.sort_values('feature').reset_index(drop=True)


output_importances_levels = output_importances_levels.sort_values('feature').reset_index(drop=True)

output_importances_levels.merge(right=final2, left_index=True, right_index=True).to_csv('output_tree_level_importances.csv', sep=',')
