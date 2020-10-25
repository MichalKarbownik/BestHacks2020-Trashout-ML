import os

files_to_change = 'plastic'
path = 'D:\PyCharm\PyCharm Projects\Wastes\dataset original merged\\' + files_to_change
file_names = os.listdir(path)

iter = 1
for filename in file_names:
    new_file_name = files_to_change + '' + str(iter) + '.jpg'
    path_to_old_file = os.path.join(path, filename)
    path_to_new_file = os.path.join(path, new_file_name)
    os.rename(path_to_old_file, path_to_new_file)
    iter += 1
