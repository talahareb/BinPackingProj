import os
import pandas as pd
from abc import ABC, abstractmethod
from instances.instance import Instance

class AbstractSolver(ABC):

    def __init__(self, inst: Instance):
        self.name = ''
        self.inst = inst
        self.sol = {
            'type_vehicle': [],
            'idx_vehicle': [],
            'id_item': [],
            'x_origin': [],
            'y_origin': [],
            'z_origin': [],
            'orient': [],
        }
        self.idx_vehicle = 0

    @abstractmethod
    def solve(self):
        pass

    def write_solution_to_file(self):
        group_ids = "353146_353145_361313"
        df_sol = pd.DataFrame.from_dict(self.sol)
        df_sol.to_csv(
            os.path.join(
                'results',
                f'sol_{self.inst.name}_{group_ids}.csv'
            ),
            index=False
        )