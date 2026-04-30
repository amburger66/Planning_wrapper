import numpy as np
import math
import torch
import torch.nn

class TreeNode():
    def __init__(self, name, depth, parent_node, children_node_guidance_scales, plan_history, guidance_scale=None, terminal_depth=None, 
            value=None, value_estimation_plan=None, virtual_visit_weight=0.0, noise_state=None, committed_plan=None):
        self.name = name
        self.depth = depth
        self._parent_node = parent_node
        self._children_node_guidance_scales = children_node_guidance_scales
        self.plan_history = plan_history
        self.guidance_scale = guidance_scale
        self.terminal_depth = terminal_depth
        self.virtual_visit_weight = virtual_visit_weight
        self.virtual_visit_count = 0
        # The maximum number of children nodes is same to the number of children node guidances
        self._children_nodes = [
            {'guidance_scale': self._children_node_guidance_scales[i], 'node': None, "virtually_visited": False}
            for i in range(len(self._children_node_guidance_scales))
        ]

        self.value = value
        self.value_estimation_plan = value_estimation_plan
        self.visit_count = 0
        self.noise_state = noise_state
        self.committed_plan = committed_plan

    def __lt__(self, other):
        return self.name < other.name
    
    def __eq__(self, other):
        return self.name == other.name

    def is_root_node(self):
        return self._parent_node is None
    
    def is_leaf_node(self):
        if self._children_nodes[0]["node"] is None:
            return False
        return True
    
    def is_expandable(self, consider_virtually_visited=False):
        if self.depth == self.terminal_depth:
            return False
        for child_node in self._children_nodes:
            if child_node["node"] is None and (not consider_virtually_visited or not child_node["virtually_visited"]):
                return True
        return False

    def is_selectable(self):
        for child_node in self._children_nodes:
            if child_node["node"] is None:
                return False
        return True

    def is_terminal(self):
        if self.depth == self.terminal_depth:
            return True
        return False
    
    def set_value(self, value):
        self.value = value

    def set_plan_history(self, plan_history):
        self.plan_history.append(plan_history)
    
    def set_value_estimation_plan(self, value_estimation_plan):
        self.value_estimation_plan = value_estimation_plan

    # Following UCT(Upper Confidence Boundary of Tree)
    def select(self, exp_weight=math.sqrt(2), leaf_parallelization=False):
        for child_node in self._children_nodes:
            if child_node["node"] is None:
                raise ValueError('Child node is None in select method')
        total_visit_count = sum([child_node["node"].visit_count + child_node["node"].virtual_visit_count * self.virtual_visit_weight for child_node in self._children_nodes])
        uct_values = []
        for child_node in self._children_nodes:
            _node = child_node["node"]
            _value = _node.value
            _visit_count = _node.visit_count + _node.virtual_visit_count * self.virtual_visit_weight
            uct_values.append(_value + exp_weight * np.sqrt(np.log(1e-6 + total_visit_count) / (1e-6 + _visit_count)))
        selected_index = np.argmax(uct_values)
        if leaf_parallelization:
            self._children_nodes[selected_index]["node"].virtual_visit_count += len(self._children_nodes)
        else:
            self._children_nodes[selected_index]["node"].virtual_visit_count += 1
        return self._children_nodes[selected_index]["node"]
    
    def expand(self, **kwargs):
        if len(kwargs) > 0:
            selected_index = int(kwargs['name'].split('-')[-1])
            assert self.name == kwargs['parent_node'].name, "Parent node is the same as the current node"
            # Create a new child node
            self._children_nodes[selected_index]['node'] = TreeNode(**kwargs)
            self._children_nodes[selected_index]['virtually_visited'] = False # virtually visit flag is False when the node is created
        return self._children_nodes[selected_index]['node']

    def get_expandable_candidate(self, index=None, consider_virtually_visited=False):
        if index is None:
            remaining_children_indices = [
                i for i in range(len(self._children_nodes)) if self._children_nodes[i]['node'] is None and (not consider_virtually_visited or not self._children_nodes[i]['virtually_visited'])
            ]
            #selected_index = remaining_children_indices[0] # To check the best-of-n MCTD
            selected_index = np.random.choice(remaining_children_indices)
        else:
            selected_index = index
        candidate_info = {
            'name': self.name + f'-{selected_index}',
            'depth': self.depth + 1,
            'parent_node': self,
            'children_node_guidance_scales': self._children_node_guidance_scales,
            'plan_history': self.plan_history.copy(),
            'guidance_scale': self._children_nodes[selected_index]['guidance_scale'],
            'value': None,
            'value_estimation_plan': None,
            'virtual_visit_weight': self.virtual_visit_weight
        }
        # this function call means that the node is virtually visited in parallel search
        self._children_nodes[selected_index]['virtually_visited'] = True
        return candidate_info

    def get_expandable_node_names(self, consider_virtually_visited=True):
        expandable_nodes = []
        for child_idx, child_node in enumerate(self._children_nodes):
            if child_node["node"] is None and (not consider_virtually_visited or not child_node["virtually_visited"]):
                expandable_nodes.append(self.name + f'-{child_idx}')
            else:
                if child_node["node"] is not None:
                    expandable_nodes.extend(child_node["node"].get_expandable_node_names(consider_virtually_visited=consider_virtually_visited))
        return expandable_nodes

    def backpropagate(self):
        self.visit_count += 1
        self.virtual_visit_count = 0 # reset the virtual visit count
        self.value = max([child_node["node"].value for child_node in self._children_nodes if child_node["node"] is not None])
        if self._parent_node is not None:
            self._parent_node.backpropagate()

    def check_virtual_visit_count(self):
        for child_node in self._children_nodes:
            if child_node["node"] is not None:
                if child_node["node"].virtual_visit_count > 0:
                    raise ValueError(f"Virtual visit count of {child_node['node'].name} is {child_node['node'].virtual_visit_count}")
                child_node["node"].check_virtual_visit_count()