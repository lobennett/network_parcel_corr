"""Default construct-to-contrast mappings for cognitive constructs."""

CONSTRUCT_TO_CONTRAST_MAP = {
    'Active Maintenance': [
        'task-nBack_contrast-match-mismatch',
        'task-nBack_contrast-twoBack-oneBack',
    ],
    'Flexible Updating': [
        'task-cuedTS_contrast-cue_switch_cost',
        'task-spatialTS_contrast-cue_switch_cost',
    ],
    'Monitoring': [
        'task-nBack_contrast-high-load-low-load',
        'task-nBack_contrast-match-mismatch',
    ],
    'Interference Control': [
        'task-flanker_contrast-incongruent-congruent',
        'task-directedForgetting_contrast-neg-con',
    ],
    'Goal Selection': [
        'task-cuedTS_contrast-cue_switch_cost',
        'task-spatialTS_contrast-cue_switch_cost',
        'task-stopSignal_contrast-go',
        'task-goNogo_contrast-go',
    ],
    'Updating Representation and Maintenance': ['task-nBack_contrast-match-mismatch'],
    'Response Selection': [
        'task-flanker_contrast-incongruent-congruent',
        'task-stopSignal_contrast-go',
        'task-goNogo_contrast-go',
    ],
    'Inhibition Suppression': [
        'task-stopSignal_contrast-stop_success',
        'task-stopSignal_contrast-stop_success-go',
        'task-stopSignal_contrast-stop_success-stop_failure',
        'task-goNogo_contrast-nogo',
        'task-directedForgetting_contrast-pos-neg',
    ],
    'Task Coordination': [
        'task-cuedTS_contrast-task_switch_cue_switch-task_stay_cue_stay',
        'task-spatialTS_contrast-task_switch_cue_switch-task_stay_cue_stay',
    ],
    'Task Baseline': [
        'task-cuedTS_contrast-task-baseline',
        'task-directedForgetting_contrast-task-baseline',
        'task-flanker_contrast-task-baseline',
        'task-goNogo_contrast-task-baseline',
        'task-nBack_contrast-task-baseline',
        'task-shapeMatching_contrast-task-baseline',
        'task-spatialTS_contrast-task-baseline',
        'task-stopSignal_contrast-task-baseline',
    ],
    'Response Time': [
        'task-cuedTS_contrast-response_time',
        'task-directedForgetting_contrast-response_time',
        'task-flanker_contrast-response_time',
        'task-goNogo_contrast-response_time',
        'task-nBack_contrast-response_time',
        'task-shapeMatching_contrast-response_time',
        'task-spatialTS_contrast-response_time',
        'task-stopSignal_contrast-response_time',
    ],
}
