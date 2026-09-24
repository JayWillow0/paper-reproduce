from __future__ import annotations

import moduletr.events as events
from moduletr import ModuleConfig, TopologyConfig
from moduletr.model import initialize_parameters


def test_event_functions_share_one_segment_local_vector_cache(monkeypatch) -> None:
    parameters, state = initialize_parameters(
        ModuleConfig(topology=TopologyConfig(n_cells=5, holders="both"), t_end_s=60.0)
    )
    calls = 0
    original = events.normalized_event_values

    def counted(time_s, current_state, current_parameters):
        nonlocal calls
        calls += 1
        return original(time_s, current_state, current_parameters)

    monkeypatch.setattr(events, "normalized_event_values", counted)
    functions = events.build_event_functions(parameters)
    assert len(functions) == 9
    assert [function.event_index for function in functions] == events.active_event_indices(parameters)
    for function in functions:
        function(0.25, state)
    assert calls == 1

    events.apply_event_update(parameters, 10.0, state, [1])
    next_functions = events.build_event_functions(parameters)
    for function in next_functions:
        function(10.0, state)
    assert calls == 2


def test_reported_event_id_does_not_depend_on_simultaneous_tolerance() -> None:
    parameters, state = initialize_parameters(
        ModuleConfig(topology=TopologyConfig(n_cells=1, holders="none"), t_end_s=20.0)
    )
    ids = events.collect_event_ids(9.0, state, parameters, [0], tolerance=0.0)
    assert ids == [1]
