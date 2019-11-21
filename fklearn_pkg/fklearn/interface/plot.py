import pandas as pd
from os.path import dirname, join

from bokeh.plotting import figure, output_file, show
from bokeh.layouts import layout, row, column
from bokeh.models import ColumnDataSource, Div
from bokeh.models.widgets import Slider, Select, Toggle, CheckboxGroup, Button
from bokeh.models.callbacks import CustomJS
from bokeh.io import curdoc, export_png


explanations_file=""

def load_csv_data(filestr):
    """
    Loads in the csv with our data in it, and returns it as a Pandas Dataframe
    """
    import csv

    datas = []
    attr_map = {}

    with open(filestr) as csvfile:
        csv_rows = csv.reader(csvfile, delimiter=',')

        # Map each index to its attribute name
        header = next(csv_rows)
        attr_map = { i : header[i] for i in range(2, len(header))}

        for row in csv_rows:
            model = row[0]
            hyperparams = row[1]
            preprocessor = row[2]
            postprocessor = row[3]
            
            # Create a new data point and add it to that model value
            datum = {'model': model, 'hyperparameters': hyperparams, 'preprocessor': preprocessor, 'postprocessor': postprocessor, **{ attr_map[i] : float(row[i]) for i in range(4, len(row))}}
            datas.append(datum)
    
    return pd.DataFrame(datas)


def load_explanations(filestr):
    """
    Loads in the metric explanations as a dictionary mapping strings to explanations
    """

    import json

    with open(filestr) as f:
        return json.load(f)


def create_plot(csvfile, jsonfile):
    """
    Creates and returns a scatter plot from the given data provided by the out.csv file. Each column will appear as a 
    checkbox to the left of the plot, allowing for hiding of non-optimal data points. Models may be toggled
    by clicking on the labeled buttons. As of now, three models are hard-coded (but this is to change in the
    future to make this more adaptable to general use cases).

    Args:
        csvfile (str): The path name of the csv file to load. By default, we assume that we are in the root directory and load "fklearn/test-file.csv"
    """

    explanations_file = jsonfile

    MODEL_COLORS = ['purple', 'orange', 'magenta', 'purple', 'green', 'blue']

    df = load_csv_data(csvfile)
    attributes = sorted(set(df.keys()) - {'model'} - {'hyperparameters'} - {'preprocessor'} - {'postprocessor'})
    
    # Assign a color to each model, recycling if need be
    colors = {model: MODEL_COLORS[i % len(MODEL_COLORS)] for i, model in enumerate(df['model'].unique())}

    # Create a color column and set their respective values
    df['color'] = df['model']
    df['visible'] = True
    df['optimal'] = True
    df.replace({'color': colors}, inplace=True)

    # Initialize the tooltips that will be displayed when hovering over a data point
    TOOLTIPS=[
        ("x", "@x"),
        ("y", "@y"),
        ("params", "@hyperparameters"),
        ("preprocessor", "@preprocessor"),
        ("postprocessor", "@postprocessor")
    ]

    data_source = ColumnDataSource(data={'x': [], 'y': [], 'model': [], 'color': [], 'hyperparameters': [], 'preprocessor': [], 'postprocessor': []})

    # Construct our scatter plot, receiving data from our data source with the given attributes
    p = figure(plot_height=500, plot_width=700, title="", toolbar_location=None, tooltips=TOOLTIPS, sizing_mode="scale_both")
    p.circle(x="x", y="y", color="color", source=data_source, size=12, line_color=None, alpha=1.0, legend="model")
    p.legend.location = "top_right"

    x_axis = Select(title="X Axis", options=attributes, value=attributes[0], css_classes=['bk-axis-select'])
    y_axis = Select(title="Y Axis", options=attributes, value=attributes[1], css_classes=['bk-axis-select'])

    def update():
        """
        Update the plot with specified data
        """

        filtered_df = df[(df['visible'] == True) & (df['optimal'] == True)]
        x_name = x_axis.value
        y_name = y_axis.value

        p.xaxis.axis_label = x_name
        p.yaxis.axis_label = y_name
        p.title.text = "{} data selected".format(len(filtered_df))
        data_source.data = {
            'x': filtered_df[x_name].values.astype(float),
            'y': filtered_df[y_name].values.astype(float),
            'model': filtered_df['model'].values,
            'color': filtered_df['color'].values,
            'hyperparameters': filtered_df['hyperparameters'].values,
            'preprocessor': filtered_df['preprocessor'].values,
            'postprocessor': filtered_df['postprocessor'].values
        }

    def create_toggle(model):
        """
        Creates a function that toggles the visibility of a given model on the plot
        """

        def toggle(toggled):
            df.loc[df['model'] == model, 'visible'] = toggled
            update()

        return toggle

    def dominates(p1, p2, attributes):
        """ 
        Returns true iff p1 dominates p2.
        """
        for attr in attributes:
            if p1[attr] >= p2[attr]:
                return False
        return True

    def filter_optimality(attrs):
        """
        Filter by pareto optimality
        """

        attr_values = [attributes[idx] for idx in attrs]
        df_list = list(df.iterrows())
        df['optimal'] = True

        # A data point p2 is optimal only if it is not dominated by any other point p1
        for j, p2 in df_list:
            df.at[j, 'optimal'] = all([not dominates(p1, p2, attr_values) for _, p1 in df_list]) 
        
        update()

    def save_screenshot(visible_attrs, filename='plot'):
        """
        Save a screenshot of the plot to the current directory with the specified file name. Also save a JSON file
        containing information about the data displayed in the plot
        """

        import json
        
        # First, export a png of the plot
        export_png(p, 'fklearn/interface/exports/{}.png'.format(filename))

        # Now create a dictionary of metadata pertaining to the current state of the plot
        plot_data = {'x_axis': x_axis.value, 'y_axis': y_axis.value}
        
        # Keep track of which models are visible on the plot
        all_models = df['model'].unique()
        visible_models = set(df[df['visible'] == True]['model'].unique())
        plot_data['model_visibility'] = { m : m in visible_models for m in all_models }

        # Keep track of which checkboxes were checked when we export the screenshot
        plot_data['pareto_checkboxes'] = { attributes[i] : i in visible_attrs for i in range(len(attributes)) }

        with open('output/{}.json'.format(filename), 'w') as f:
            json.dump(plot_data, f)



    # Create our toggle buttons to show/hide different models on the plot
    toggles = []
    for model in colors:
        toggle = Toggle(label="{}".format(model), button_type="success", active=True, css_classes=['bk-btn-model-{}'.format(colors[model])])
        toggle.on_click(create_toggle(model))
        toggles.append(toggle)

    x_axis.on_change('value', lambda attr, old, new: update())
    y_axis.on_change('value', lambda attr, old, new: update())

    checkbox_group = CheckboxGroup(labels=attributes, active=list(range(len(attributes))), css_classes=['bk-checkbox-group'])
    checkbox_group.on_click(lambda checked_attrs: filter_optimality(checked_attrs))
    
    screenshot_btn = Button(label="Export Plot", button_type="warning", css_classes=['screenshot-btn'])
    screenshot_btn.on_click(lambda: save_screenshot(visible_attrs=checkbox_group.active))

    # Load metric explanations as tooltips for the checkboxes
    metric_dict = load_explanations(explanations_file)
    
    inputs = column(x_axis, y_axis, *toggles, checkbox_group, screenshot_btn, width=320, height=500, sizing_mode="fixed")
    plot_row = row(inputs, p, css_classes=['layout-container'])
    
    # NOTE: Super hacky way to do this, but it was the only easy way I could fine.
    metric_tooltip_js = """var checkboxes = document.querySelectorAll('.bk-checkbox-group .bk-input-group label.bk');\n"""
    for i in range(len(attributes)):
        metric_tooltip_js += """checkboxes[{}].setAttribute('title', `{}`);\n""".format(i, metric_dict[attributes[i]])

    # Create a setTimeout wrapper around the function so the DOM has a chance to mount
    metric_tooltip_js = """setTimeout(function() {\n""" + metric_tooltip_js + """}, 200);\n"""
    explanation_callback = CustomJS(args=dict(), code=metric_tooltip_js)
    p.x_range.js_on_change('start', explanation_callback)
    checkbox_group.js_on_click(explanation_callback)

    # Initial load of our data
    filter_optimality(range(len(attributes)))
    
    return plot_row
