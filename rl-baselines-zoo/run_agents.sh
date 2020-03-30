# for seed in 1 2 3 4
# do
# 	    python train.py --algo a2c  --env su_acrobot_cdc-v0 -n 2000000 --seed $seed -f baseline_log2 &
# done
# wait

for seed in 1 2 3 4
do
	    python train.py --algo ppo2  --env su_acrobot_cdc-v0 -n 2000000 --seed $seed -f baseline_log2 &
done
wait

# for seed in 1 2 3 4
# do
# 	    python train.py --algo trpo --env su_acrobot_cdc-v0 -n 2000000 --seed $seed -f baseline_log2 &
# done
# wait

# for seed in 1 2 3 4
# do
# 	    python train.py --algo ddpg --env su_acrobot_cdc-v0 -n 2000000 --seed $seed -f baseline_log2 & 
# done
# wait

# for seed in 1 2 3 4
# do
# 	    python train.py --algo td3  --env su_acrobot_cdc-v0 -n 2000000 --seed $seed -f baseline_log2 &
# done
# wait

# for seed in 1 2 3 4
# do
# 	    python train.py --algo sac  --env su_acrobot_cdc-v0 -n 2000000 --seed $seed -f baseline_log2 &
# done
# wait

