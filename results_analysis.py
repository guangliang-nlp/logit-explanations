import json
file = open("results/winogender/winogender_gpt2-xl_ours.json",'r')
data = json.load(file)
for i in range(240):
    print(len(data["logit_aff_x_j_alti"][i]))