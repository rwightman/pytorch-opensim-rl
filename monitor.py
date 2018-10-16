import time
from baselines.bench import Monitor as _Monitor


class Monitor(_Monitor):

    def __init__(self, env, filename, allow_early_resets=False, reset_keywords=(), info_keywords=()):
        super(Monitor, self).__init__(
            env, filename,
            allow_early_resets=allow_early_resets,
            reset_keywords=reset_keywords, info_keywords=info_keywords)
        self.benchmark_rewards = []
        self.episode_benchmark_rewards = []
        if 'rb' in info_keywords:
            self.do_benchmark = True
        else:
            self.do_benchmark = False

    def reset_state(self):
        self.benchmark_rewards = []
        super(Monitor, self).reset_state()

    def update(self, ob, rew, done, info):
        if self.do_benchmark and 'rb' in info:
            self.benchmark_rewards.append(info['rb'])
        self.rewards.append(rew)
        if done:
            self.needs_reset = True
            eprewb = sum(self.benchmark_rewards)
            eprew = sum(self.rewards)
            eplen = len(self.rewards)
            epinfo = {
                "r": round(eprew, 6),
                "l": eplen,
                "t": round(time.time() - self.tstart, 6)}
            for k in self.info_keywords:
                epinfo[k] = info[k]
            if self.do_benchmark:
                epinfo["rb"] = eprewb  # overwrite with episode benchmark
            self.episode_benchmark_rewards.append(eprewb)
            self.episode_rewards.append(eprew)
            self.episode_lengths.append(eplen)
            self.episode_times.append(time.time() - self.tstart)
            epinfo.update(self.current_reset_info)
            self.results_writer.write_row(epinfo)

            if isinstance(info, dict):
                info['episode'] = epinfo

        self.total_steps += 1
