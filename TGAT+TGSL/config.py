'''
Training Configuration
'''
class Config:
    BATCH_SIZE = 200
    EPOCH = 50
    SSL_EPOCH = 10
    LR = 1e-4
    SSL_LR = 1e-4

    N_DEGREE = 20
    N_HEAD = 2
    N_LAYER = 2
    DROPOUT = 0.1
    NODE_DIM = 100
    TIME_DIM = 100
    MAX_ROUND = 8


'''
Wiki Dataset Configuration
'''
class WikiConfig(Config):
    pass


'''
Reddit Dataset Configuration
'''
class RedditConfig(Config):
    pass


'''
Escorts Dataset Configuration
'''
class ESConfig(Config):
    pass


if __name__ == '__main__':
    config = Config()

