import pickle

f = open("results/future_trace.pickle", "rb")

res = pickle.load(f)

print(res['pos'].keys())

print('ffn:')
print('pos')
print(res['pos']['R2'])
print(res['pos']['HyperParams'])
print('vel')
print(res['vel']['R2'])
print(res['vel']['HyperParams'])

# print('gru:')
# print('pos')
# print(res['gru']['pos']['R2'])
# print(res['gru']['pos']['HyperParams'])
# print('vel')
# print(res['gru']['vel']['R2'])
# print(res['gru']['vel']['HyperParams'])