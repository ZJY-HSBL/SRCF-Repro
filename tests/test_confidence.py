from pathlib import Path

import pytest

from srcf_repro.confidence import ConfidenceRecommender
from srcf_repro.confirmatory import (cohorts, evaluate_cohort, paired_interval,
                                      select_on_validation, publish, METHODS)
from srcf_repro.experiment import synthetic
from srcf_repro.model import Profile


def test_missing_rating_components_fallback_to_demographic():
    profiles={1:Profile(25,'M','dev'),2:Profile(25,'M','dev')}
    ratings={1:{1:5},2:{1:5,2:4,3:1}}
    m=ConfidenceRecommender(ratings,profiles,seeds={1,2},mode='sr',
                            lambda_support=10,prediction='normalized')
    assert m.similarity(1,2)==pytest.approx(1)
    with pytest.raises(ValueError):
        ConfidenceRecommender(ratings,profiles,mode='sr',lambda_support=-1)
    with pytest.raises(ValueError):
        ConfidenceRecommender(ratings,profiles,mode='sr',prediction='paper')


def test_seed_ablation_can_change_similarity():
    ratings={1:{1:1,2:5,3:1,4:5},2:{1:5,2:1,3:1,4:5}}
    profiles={1:Profile(25,'M','dev'),2:Profile(25,'M','dev')}
    opts=dict(ratings=ratings,profiles=profiles,seeds={1,2},mode='sr',
              prediction='normalized',lambda_support=2)
    both=ConfidenceRecommender(**opts)
    ablated=ConfidenceRecommender(**opts,use_seed=False)
    assert both.similarity(1,2)!=ablated.similarity(1,2)
    assert 1<=both.predict(1,4,2)<=5


def test_cohorts_and_train_only_selection(tmp_path):
    ratings,profiles=synthetic(123,users=48,items=38)
    params=dict(val_n=4,test_n=5,prior_batches=2,prior_per_batch=3,
                prior_seed_batches=2,prior_seed_batch_size=2,seed=314159)
    ref,val,test,prior=cohorts(ratings,**params)
    ref2,v2,t2,p2=cohorts(ratings,**params)
    assert (ref,val,test,prior)==(ref2,v2,t2,p2)
    assert prior==10
    assert len(val)==4 and len(test)==5
    assert set(ref).isdisjoint({u for u,_,_ in val+test})
    assert set(u for u,_,_ in val).isdisjoint(u for u,_,_ in test)
    for _,hold,seen in val+test:
        assert set(hold).isdisjoint(seen)
    vrows=evaluate_cohort(ref,val,ratings,profiles)
    sel=select_on_validation(vrows)
    assert sel['selected'] in ('confidence_2','confidence_10','confidence_30')
    trows=evaluate_cohort(ref,test,ratings,profiles,partition='test')
    assert len(trows)==5*5*len(METHODS)
    paired=paired_interval(trows,sel['selected'],'sr',draws=100)
    assert paired['n_users']==5
    publish(vrows,trows,sel,tmp_path/'results',source='SYNTHETIC',
            prior_excluded=prior,reference_users=len(ref),seed=314159)
    assert (tmp_path/'results'/'holdout_mae.svg').exists()
    assert (Path('docs')/'CONFIDENCE_STUDY.md').exists()


def test_val_selection_requires_complete_predeclared_scores():
    with pytest.raises(ValueError):
        select_on_validation([{'method':'confidence_2','observed':3,'mae':1.0}])
