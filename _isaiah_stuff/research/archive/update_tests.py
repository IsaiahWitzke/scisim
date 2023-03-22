import os

folder_name = 'grid_v2'

for f in os.listdir(os.path.join(os.path.dirname(__file__), '..', folder_name)):

    with open(os.path.join(os.path.dirname(__file__), '..', folder_name, f), 'r') as file:
        filedata = file.read()

    # Replace the target string

    filedata = filedata.replace('tol="1.0e-6"', 'tol="1.0e-9"')

    # Write the file out again
    with open(os.path.join(os.path.dirname(__file__), '..', folder_name, f), 'w') as file:
        file.write(filedata)


