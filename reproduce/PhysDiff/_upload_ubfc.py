"""把 PhysDiff 代码和 UBFC h5 传到 AutoDL。密码从环境变量 SSH_PASS 读取。"""

import os
import sys

import paramiko

HOST = "connect.cqa1.seetacloud.com"
PORT = 34300
LOCAL_REPRO = r"d:\maxzhang\python files\RePSS\reproduce"
LOCAL_UBFC = r"D:\maxzhang\python files\datasets\ubfc_h5"
REMOTE_REPRO = "/root/autodl-tmp/RePSS/reproduce"
REMOTE_UBFC = "/root/autodl-tmp/ubfc_h5"

CODE_FILES = [
    "PhysDiff/train_local.py",
    "PhysDiff/config.py",
    "PhysDiff/model/__init__.py",
    "PhysDiff/model/physdiff.py",
    "PhysDiff/model/hybrid_denoiser.py",
    "common/train_viz.py",
    "CanonicalPhys/evaluation/__init__.py",
    "CanonicalPhys/evaluation/post_process.py",
    "CanonicalPhys/evaluation/metrics.py",
]


def mkdir_p(sftp, path):
    parts = path.strip("/").split("/")
    current = ""
    for part in parts:
        current += "/" + part
        try:
            sftp.stat(current)
        except FileNotFoundError:
            sftp.mkdir(current)


def main():
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        HOST,
        port=PORT,
        username="root",
        password=os.environ["SSH_PASS"],
        timeout=30,
        banner_timeout=30,
    )
    client.get_transport().set_keepalive(30)
    sftp = client.open_sftp()
    for relative in CODE_FILES:
        remote = REMOTE_REPRO + "/" + relative.replace("\\", "/")
        mkdir_p(sftp, os.path.dirname(remote))
        local = os.path.join(LOCAL_REPRO, relative.replace("/", os.sep))
        print("code", relative, flush=True)
        sftp.put(local, remote)
    mkdir_p(sftp, REMOTE_UBFC)
    names = sorted(name for name in os.listdir(LOCAL_UBFC) if name.endswith(".h5"))
    for index, name in enumerate(names, 1):
        remote = REMOTE_UBFC + "/" + name
        local = os.path.join(LOCAL_UBFC, name)
        try:
            if sftp.stat(remote).st_size == os.path.getsize(local):
                print(f"skip {index}/{len(names)} {name}", flush=True)
                continue
        except FileNotFoundError:
            pass
        print(f"data {index}/{len(names)} {name} {os.path.getsize(local)}", flush=True)
        sftp.put(local, remote)
    sftp.close()
    client.close()
    print("upload done", len(names), flush=True)


if __name__ == "__main__":
    sys.exit(main())
