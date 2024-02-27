import numpy as np
import os

f = open('/home/user/Desktop/log_qatm_scoreOnGPU_nonorm_init_in_qatm_model.txt', 'r')
lines = f.readlines()
f.close()


start = 0
end = 0

req_info = 'target featex time in secs.'



whole_time = []
batched_time = []
one_sample_time = []


def extract(list_txts, start_index, end_index, info):
    if start_index == 0:
        for indx, line_ in enumerate(list_txts):
            if line_.startswith('  0%|          '):
                start_index = indx
                break
        temp = list_txts[start_index:end_index]
        temp[0] = temp[0].split(']')[1]
    else:
        temp = list_txts[start_index + 1:end_index]
    
    eisum_counter = 0
    batch_or_not = False
    returned_info = []
    for line_r in temp:
        try:
            identifier, value = line_r.split(':  ')
            if 'eisum' in identifier:
                eisum_counter += 1
            if identifier == info:
                returned_info.append(float(value[:-1]))
        except:
            print(line_r)
            pass
    if eisum_counter > 1:
        batch_or_not = True

    
    return returned_info, batch_or_not, end_index


for index, line in enumerate(lines):
    if line.startswith('*'):
        end = index
        extracted_info, batched, start = extract(lines, start, end, req_info)
        whole_time.extend(extracted_info)
        if batched:
            batched_time.extend(extracted_info)
        else:
            one_sample_time.extend(extracted_info)


print('Whole average time for {} in sec.:', np.sum(np.array(whole_time)) / len(whole_time))
print('batched average time for {} in sec.:', np.sum(np.array(batched_time)) / len(batched_time))
print('One sample average time for {} in sec.:', np.sum(np.array(one_sample_time)) / len(one_sample_time))






