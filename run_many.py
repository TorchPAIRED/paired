# python3 run_many_runs rootdir num_runs
import subprocess
import sys


def doit(root_dir, num_runs):
    import subprocess, os
    my_env = os.environ.copy()
    my_env["PATH"] = "/usr/sbin:/sbin:" + my_env["PATH"]

    for run_id in range(int(num_runs)):
        import io
        
        argsss = f"""
       --xpid=ued-MultiGrid-GoalLastAdversarial-v0-paired-lstm256ae-lr0.0001-epoch5-mb1-v0.5-henv0.0-ha0.0-tl_0 
--env_name=MultiGrid-GoalLastAdversarial-v0 
--use_gae=True 
--gamma=0.995 
--gae_lambda=0.95 
--seed={run_id*100} 
--recurrent_arch=lstm 
--recurrent_agent=True 
--recurrent_adversary_env=True 
--recurrent_hidden_size=256 
--lr=0.0001 
--num_steps=256 
--num_processes=32 
--num_env_steps=1000000000 
--ppo_epoch=5 
--num_mini_batch=1 
--entropy_coef=0.0 
--value_loss_coef=0.5 
--clip_param=0.2 
--clip_value_loss=True 
--adv_entropy_coef=0.0 
--algo=ppo 
--ued_algo=paired 
--log_interval=10 
--screenshot_interval=1000 
--log_grad_norm=True 
--handle_timelimits=True 
--test_env_names=MultiGrid-SixteenRooms-v0,MultiGrid-Labyrinth-v0,MultiGrid-Maze-v0 
--log_dir={root_dir}/{run_id} 
--log_action_complexity=True 
--checkpoint=True 
        """.replace("\n", " ")

        try:
            proc = subprocess.Popen(f"python3 -m train {argsss}", env=my_env, shell=True, stdout=subprocess.PIPE)
            for line in io.TextIOWrapper(proc.stdout, encoding="utf-8"):
                print(line)
        except:
            try:
                import signal
                os.killpg(proc.pid, signal.SIGKILL)
                #proc.kill()
            except:
                pass
            print("PROCESS DIED. EXITING NOW.")
            exit(1)

if __name__ == "__main__":
    doit(sys.argv[1], sys.argv[2])