import pandas as pd

c4_names = ['S' + str(i) for i in range(42)]
c4_names.append( 'class')
c4_types = ['S1' for i in range(42)]
c4_types.append( 'S4')
# print(c4_names, len(c4_names), len(c4_types))
connect = pd.read_csv('../datasets/connect/connect-4.data', sep=',', header=None)

# print(connect.values)
connect = connect.replace('x', 0)
connect = connect.replace('o', 1)
connect = connect.replace('b', 2)

connect = connect.replace('win', 0)
connect = connect.replace('loss', 1)
connect = connect.replace('draw', 2)
connect.to_csv('../datasets/connect/connect_norm1.data.csv',index=False,header=False)