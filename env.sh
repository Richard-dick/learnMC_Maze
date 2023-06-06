update_python(){
    # 首先更新list
    echo "apt-get update"
    apt-get update
    echo "update over !!!!!!!!!!!!!!"


    echo "apt-get upgrade"
    apt-get upgrade
    echo "update over !!!!!!!!!!!!!!"


    # 配置python环境
    update-alternatives --install /usr/bin/python python /usr/bin/python2.7 1
    update-alternatives --install /usr/bin/python python /usr/bin/python3.6 2

    # 并安装3.9

    ## dependency
    # apt install build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libreadline-dev libffi-dev libsqlite3-dev wget libbz2-dev

    ## way-2:
    apt update
    apt install software-properties-common
    apt update
    apt install python3.8

    update-alternatives --install /usr/bin/python python /usr/bin/python3.8 3

    # 配置pip
    ## dependency
    apt-get install python3.8-distutils
    curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
    python get-pip.py
}


setup_git(){
    # 克隆仓库
    apt-get install git-lfs
    git clone https://github.com/seanmperkins/bci-decoders.git
    cd bci-decoders/
    apt install git-lfs
    git lfs install
    git lfs pull
    
    cd ..
    git clone https://github.com/Richard-dick/learnMC_Maze.git

    cd learnMC_Maze/
    pwd
    git config --global user.email "you@example.com"
    git config --global user.name "Your Name"

    pip install -r requirement

    cd data/

}


# setup_git