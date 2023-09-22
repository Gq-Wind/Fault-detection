import subprocess


def install_dependencies():
    try:
        subprocess.check_call(['pip', 'install', '-r', 'requirements.txt'])
        print("依赖项安装完成！")
    except subprocess.CalledProcessError:
        print("依赖项安装失败。")


if __name__ == '__main__':
    install_dependencies()
