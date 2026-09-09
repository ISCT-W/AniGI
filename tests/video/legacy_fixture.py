"""Seed pre-contract snapshots to test backwards compatibility, not a public init path."""
from pathlib import Path
from anigen.video import runtime


def seed_legacy(run_id, plan, runs):
    runtime.validate_plan(plan)
    folder = Path(runs) / runtime.ident(run_id)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / 'state.json'
    runtime.require(not path.exists(), 'run already exists')
    state = {'version':1,'id':run_id,'plan':plan,'authorization':None,'evidence':{},
             'attempts':[],'accepted':{},'final':None,'events':[]}
    if plan.get('production_contract'):
        state['production_contract_sha256'] = runtime.directing.fingerprint(plan['production_contract'])
    runtime.event(state,'created')
    runtime.save(path,state)
    return state
