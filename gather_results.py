import os
import re
import torch
import os.path as op


p = re.compile(r'\[Test(.+?)-(.+?)\]')
lr_list = [1e-5, 2e-5, 3e-5, 4e-5, 5e-5]

data_info = 'out/mvsa-s'
train_info = '[s1][t1]'
temp_info = [
    '',
    '[nf_resnet50-1]',
    '[nf_resnet50-2]',
    '[nf_resnet50-3]',
    '[nf_resnet50-4]',
    '[nf_resnet50-5]',
]

for temp in temp_info:
    root = os.path.join(data_info, train_info+temp)

    results = []
    for lr in lr_list:
        path = op.join(root, str(lr))
        file_list = os.listdir(path)
        txt_files = [f for f in file_list if f[-4:] == '.txt']
        
        result = [p.search(s).groups() for s in txt_files]
        result = [[float(i) for i in tup] for tup in result]
        result = torch.tensor(result)
        
        max_result, _ = torch.max(result, dim=0)
        mean_result = torch.mean(result, dim=0)
        std = torch.std(result, dim=0)
        result = torch.cat([max_result, mean_result, std], dim=0)
        results.append(result)

    results = torch.stack(results, dim=0)
    final_result, order = torch.max(results, dim=0)
    std_mean = torch.mean(results[:,4:6], dim=0)
    assert final_result[0] == results[order[0]][0]
    final_result[4] = results[order[2]][4]
    final_result[5] = results[order[3]][5]
    final_result = torch.cat([final_result, std_mean], dim=0)

    max_res = '_'.join(['{:.2f}'.format(i.item()) for i in final_result[0:2]])
    mean_res = '_'.join(['{:.2f}'.format(i.item()) for i in final_result[2:4]])
    std = '_'.join(['{:.2f}'.format(i.item()) for i in final_result[4:6]])
    mean_std = '_'.join(['{:.2f}'.format(i.item()) for i in final_result[6:8]])
    file_name = '__'.join([max_res, mean_res, std, mean_std]) + '.txt'

    with open(op.join(root, file_name), 'w', encoding='utf-8') as f:
        f.write(str(order) + '\n')
        f.write('|       Max       |       Mean      |      Std       |\n')
        f.write('|   Acc  | Mac-F1 |   Acc  | Mac-F1 |  Acc  | Mac-F1 |\n')
        for line in results:
            f.write('|')
            for i in line:
                f.write('  {:.2f}  '.format(i.item()))
            f.write('|\n')
