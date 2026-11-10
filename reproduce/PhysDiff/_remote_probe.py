import os
import sys

import paramiko

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(
    "connect.cqa1.seetacloud.com",
    port=34300,
    username="root",
    password=os.environ["SSH_PASS"],
    timeout=30,
    banner_timeout=30,
)
command = sys.stdin.read()
stdin, stdout, stderr = client.exec_command(command, timeout=120)
out = stdout.read().decode("utf-8", "replace")
err = stderr.read().decode("utf-8", "replace")
sys.stdout.write(out)
if err.strip():
    sys.stdout.write("\nSTDERR\n" + err)
client.close()
