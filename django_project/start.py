import os
import subprocess
import sys


def start_django_server():
    try:
        # 启动Django后端
        subprocess.check_call(['python', 'manage.py', 'runserver'])
        print("Django后端已启动！")
    except subprocess.CalledProcessError:
        print("启动Django后端时出错。")


if __name__ == '__main__':
    p1 = os.path.abspath(sys.argv[0])
    print(p1)

    name1 = os.path.dirname(p1)
    print(name1)
    name2 = os.path.dirname(name1)
    # 切换到项目根目录
    os.chdir(name2)
    start_django_server()
