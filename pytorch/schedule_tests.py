EPS = 1e-8


def test_schedule(schedule_type,
                  beta_start=5,
                  beta_end=14,
                  epochs=20,
                  cycles=1,
                  verbose=False):
    from train_feature_vae import BetaSchedule

    beta0 = 0.1
    beta1 = 1.0

    schedule = BetaSchedule(schedule_type, beta0, beta1,
                            beta_start, beta_end, cycles)
    for epoch in range(epochs):
        beta = schedule.get_beta(epoch)
        if verbose:
            print('Schedule {}, epoch {}: beta {}'.format(schedule_type,
                                                          epoch,
                                                          beta))
        if epoch < beta_start:
            assert abs(beta - beta0) < EPS
        elif epoch > beta_end:
            assert abs(beta - beta1) < EPS
        else:
            assert beta0 - EPS <= beta <= beta1 + EPS
