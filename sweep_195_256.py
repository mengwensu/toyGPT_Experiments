import random
import subprocess
import re
import os

def generate_random_string():
    return ''.join(random.choice('ABCD') for _ in range(3))

def get_script():
    with open("run_script.txt", 'w') as f:
        for i in range(100):
            start = generate_random_string()
            s = f'python3 sample.py --out_dir=out-shakespeare-char --start="{start}" --num_samples=100 --max_new_tokens=1 --device=cuda'
            f.write(s + '\n')


# Check if the string is the correct or wrong output
def checkString(str_list):

    correct = 0
    wrong = 0
    for s in str_list:
        if s[-1] != '\n':
            correct += 1 
        else:
            wrong += 1

    return correct / 100

def read_script(file):
    accuracy_list = []
    ind = 0
    err = 0

    with open(file, 'r') as f:
        commands = f.readlines()

    for command in commands:
        print(f"Executing command: {command.strip()}")
        try:
            # Execute the command
            result = subprocess.Popen(command.strip(), stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, text=True)
            stdout, stderr = result.communicate()

            if result.returncode != 0:
                err = 1
                print(f"Error executing command: {stderr}")
                continue

            # Debug print the stdout to see what is being parsed
            print(f"Command output:\n{stdout}")

            # Parse the relevant information
            matches = re.findall(r'([A-Z]+)\n-{7}', stdout)
            
            print(f"Index: {ind}")
            ind += 1
            print(f"Matches: {matches}")

            # Check the accuracy
            accuracy = checkString(matches)
            print(f"Accuracy: {accuracy}")
            accuracy_list.append(accuracy)
        except Exception as e:
            print(f"Exception occurred: {e}")

    # Calculate average accuracy
    average_accuracy = sum(accuracy_list) / len(accuracy_list) if accuracy_list else 0
    return err, accuracy_list, average_accuracy

def get_new_training_set(dataset, size):
    with open(dataset, 'r') as f:
        data = f.readlines()

    new_training_set = random.sample(data, size)
    # save new training set to inpu.txt
    with open('data/shakespeare_char/input.txt', 'w') as f:
        for i in new_training_set:
            f.write(i)


# get_new_training_set("data/shakespeare_char/full_dataset.txt",0)
# read_script("run_script.txt")

def print_real_time_output(process):
    # Ensure that output from the process is processed correctly
    for line in iter(process.stdout.readline, ''):
        print(line.strip())
    for line in iter(process.stderr.readline, ''):
        print(line.strip())


# get_new_training_set("data/shakespeare_char/full_dataset.txt", 256)


for i in range(194, 257):
    avg_acc = []
    for j in range(2):
        print("ROUND", j+1)
        print('Data Size: ', i)
        get_new_training_set('data/shakespeare_char/full_dataset.txt', i)
        get_script()
        print('after get script')
        # Run prepare.py
        command = 'python3 data/shakespeare_char/prepare.py'
        prepare_run = subprocess.run(command.strip(), shell=True, check=True, capture_output=True, text=True)
        print(prepare_run.stdout)
        print('after prepare')

        # If block size if less than default 32, find a smaller block size 
        if i <= 64:
            # block_size = i / 2 -1
            # blk_str = '--block_size=' + block_size
            # command = ['python3', 'train.py', 'config/train_shakespeare_char.py', blk_str]
            # print('block size:', block_size)
            # sample_run1 = subprocess.run(command.strip(), shell=True, check=True, capture_output=True, text=True)
            # print(sample_run1.stdout)

            # Adjust the block size``

            block_size = i // 2 - 1
            blk_str = f'--block_size={block_size}' 
            print('block size:', block_size)
            command = ['python3', 'train.py', 'config/train_shakespeare_char.py', blk_str]
            # Start the subprocess and capture its output
            train_run1 = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            
            # Print the output in real-time
            print_real_time_output(train_run1)
            train_run1.wait()
        else:
            # command = ['python3', 'train.py', 'config/train_shakespeare_char.py', '--block_size=32']
            # print('block size:', 32)
            # sample_run2 = subprocess.run(command.strip(), shell=True, check=True, capture_output=True, text=True)
            # print(sample_run2.stdout)

            command = ['python3', 'train.py', 'config/train_shakespeare_char.py', '--block_size=32']
            print('block size:', 32)
            train_run2 = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            
            # Print the output in real-time
            print_real_time_output(train_run2)
            train_run2.wait()
            
        # Read accuracy from run_script.txt
        print('before run script')
        err, _, acc = read_script('run_script.txt')
        print('after run script')

        print(i, "round", j+1, "avg acc:", acc)
        avg_acc.append(acc)

        # After done with the testing the samples for the current trained dataset, delete ckpt.pkl file
        try:
            os.remove('out-shakespeare-char/ckpt.pt')
            print("Deleted ckpt.pt")
        except FileNotFoundError:
            print("File ckpt.pt not found")
            

    # Write results to log.txt
    print('writing to log')
    overall_acc = sum(avg_acc) / 2
    print(i, "Overall Average Accuracy:", overall_acc )
    with open('log_194_256.txt', 'a') as f:
        if err == 1:
            l = f"{i},{overall_acc} ckpt error\n"
        else:
            l = f"{i},{overall_acc}\n"
        f.write(l)


