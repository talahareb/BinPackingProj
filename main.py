from instances import Instance
from solver_353146_353145_361313 import solver_353146_353145_361313

if __name__ == '__main__':

    dataset_name = 'Dataset9'

    inst = Instance(dataset_name)

    solver = solver_353146_353145_361313(inst)

    solver.solve()