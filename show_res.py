import pickle

f = open("results/test-2.pickle", "rb")

res = pickle.load(f)

print(res['ffn']['pos'].keys())

print('ffn:')
print('pos')
print(res['ffn']['pos']['R2'])
print(res['ffn']['pos']['HyperParams'])
print('vel')
print(res['ffn']['vel']['R2'])
print(res['ffn']['vel']['HyperParams'])

print('gru:')
print('pos')
print(res['gru']['pos']['R2'])
print(res['gru']['pos']['HyperParams'])
print('vel')
print(res['gru']['vel']['R2'])
print(res['gru']['vel']['HyperParams'])