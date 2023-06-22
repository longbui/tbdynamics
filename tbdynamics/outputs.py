
from summer2.parameters import DerivedOutput

def request_flow_output(model, output_name, flow_name, save_results=True):
    model.request_output_for_flow(output_name, flow_name, save_results=save_results)

def request_aggregation_output(model, output_name, sources, save_results=True):
    model.request_aggregate_output(output_name, sources, save_results=save_results)

def request_normalise_flow_output(model, output_name, source, save_results=True):
    model.request_function_output(output_name, DerivedOutput(source) / model.timestep, save_results=save_results)