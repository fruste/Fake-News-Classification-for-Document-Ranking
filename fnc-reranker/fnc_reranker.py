from numpy import dot
from pyserini.search import pysearch
import numpy as np


def make_run_file(file, topics, searcher, w_bm25, w_rnp):
    probTrue = np.load('trueProbs_d2v.npy',allow_pickle='TRUE').item()
    with open(file, 'w') as runfile:
        cnt = 0
        print('Running {} queries in total'.format(len(topics)))
        for id in topics:
            query = topics[id]['title'].encode('utf-8')
            hits = searcher.search(query, 10)
            for i in range(0, len(hits)):
                doc_id = hits[i].docid

                bm25_score = hits[i].score
                real_news_prob = probTrue[str(doc_id)]

                score = w_bm25 * bm25_score + w_rnp * real_news_prob

                _ = runfile.write('{} Q0 {} {} {:.6f} Anserini\n'.format(id, hits[i].docid, i+1, score))
                cnt += 1
                if cnt % 100 == 0:
                	print(f'{cnt} queries completed')

if __name__ == "__main__":
	topics = pysearch.get_topics('robust04')
	searcher = pysearch.SimpleSearcher('robust_index')


	make_run_file('run.fnc-reranker.txt', topics , searcher, 0.5, 0.5)
