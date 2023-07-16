import pickle

f = open("results/32-8-trace&&32pos/res.pickle", "rb")

res = pickle.load(f)

print(res.keys())

for var_key in res.keys():
    print(str(var_key))
    for sp in res[var_key].keys():
        print(str(sp))
        print(res[var_key][sp]['MSE'])

# print('pos:')
# print(res['pos']['R2'])
# print(res['pos']['HyperParams'])
# print('vel')
# print(res['vel']['R2'])
# print(res['vel']['HyperParams'])

# print('gru:')
# print('pos')
# print(res['gru']['pos']['R2'])
# print(res['gru']['pos']['HyperParams'])
# print('vel')
# print(res['gru']['vel']['R2'])
# print(res['gru']['vel']['HyperParams'])