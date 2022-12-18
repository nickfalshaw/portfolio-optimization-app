from ._anvil_designer import MainTemplate
from anvil import *
import anvil.tables as tables
import anvil.tables.query as q
from anvil.tables import app_tables
import anvil.server

class Main(MainTemplate):
  def __init__(self, **properties):
    # Set Form properties and Data Bindings.
    self.init_components(**properties)

    # Any code you write here will run when the form opens.
    self.method_dropdown.items = ["Variational Autoencoder (VAE)", "Principal Component Analysis (PCA)",
                                 "Agglomerative Clustering (AC)", "Affinity Propagation Clustering (AP)"]
 
  def link_1_click(self, **event_args):
    """This method is called when the link is clicked"""
    from ..AffinityPropagation import AffinityPropagation
    new_panel = AffinityPropagation()
    get_open_form().content_panel_copy.clear()
    get_open_form().content_panel_copy.add_component(new_panel)

  def link_2_click(self, **event_args):
    """This method is called when the link is clicked"""
    from ..AgglomerativeClustering import AgglomerativeClustering
    new_panel = AgglomerativeClustering()
    get_open_form().content_panel_copy.clear()
    get_open_form().content_panel_copy.add_component(new_panel)

  def link_3_click(self, **event_args):
    """This method is called when the link is clicked"""
    from ..PrincipleComponentAnalysis import PrincipleComponentAnalysis
    new_panel = PrincipleComponentAnalysis()
    get_open_form().content_panel_copy.clear()
    get_open_form().content_panel_copy.add_component(new_panel)

  def link_4_click(self, **event_args):
    """This method is called when the link is clicked"""
    from ..BayesianAutoencoder import BayesianAutoencoder
    new_panel = BayesianAutoencoder()
    get_open_form().content_panel_copy.clear()
    get_open_form().content_panel_copy.add_component(new_panel)

  def link_5_click(self, **event_args):
    """This method is called when the link is clicked"""
    from ..RunOptimization import RunOptimization
    new_panel = RunOptimization()
    get_open_form().content_panel_copy.clear()
    get_open_form().content_panel_copy.add_component(new_panel)
    
  def link_6_click(self, **event_args):
    """This method is called when the link is clicked"""
    from ..Home import Home
    new_panel = Home()
    get_open_form().content_panel_copy.clear()
    get_open_form().content_panel_copy.add_component(new_panel)

  def link_7_click(self, **event_args):
    """This method is called when the link is clicked"""
    from ..AffinityPropagation import AffinityPropagation
    new_panel = AffinityPropagation()
    get_open_form().content_panel_copy.clear()
    get_open_form().content_panel_copy.add_component(new_panel)

  def link_8_click(self, **event_args):
    """This method is called when the link is clicked"""
    from ..AgglomerativeClustering import AgglomerativeClustering
    new_panel = AgglomerativeClustering()
    get_open_form().content_panel_copy.clear()
    get_open_form().content_panel_copy.add_component(new_panel)

  def link_9_click(self, **event_args):
    """This method is called when the link is clicked"""
    from ..PrincipleComponentAnalysis import PrincipleComponentAnalysis
    new_panel = PrincipleComponentAnalysis()
    get_open_form().content_panel_copy.clear()
    get_open_form().content_panel_copy.add_component(new_panel)

  def link_10_click(self, **event_args):
    """This method is called when the link is clicked"""
    from ..VariationalAutoencoder import VariationalAutoencoder
    new_panel = VariationalAutoencoder()
    get_open_form().content_panel_copy.clear()
    get_open_form().content_panel_copy.add_component(new_panel)

  def link_11_click(self, **event_args):
    """This method is called when the link is clicked"""
    from ..RunOptimization import RunOptimization
    new_panel = RunOptimization()
    get_open_form().content_panel_copy.clear()
    get_open_form().content_panel_copy.add_component(new_panel)

  def button_1_click(self, **event_args):
    """This method is called when the button is clicked"""
    from ..RunOptimization import RunOptimization
    new_panel = RunOptimization()
    get_open_form().content_panel_copy.clear()
    get_open_form().content_panel_copy.add_component(new_panel)

  def optimize_button_click(self, **event_args):
    """This method is called when the button is clicked"""
    self.returns_label.visible = False
    self.volatility_label.visible = False
    self.sim_label.visible = False
    self.expected_returns.visible = False
    self.volatility.visible = False
    self.sim.visible = False
    self.efficient_frontier_plot.visible = False
    self.optimal_weights_grid.visible = False
    self.error_message.visible = False
    self.response.visible = False
    self.optimize_button.visible = False
    self.pie_chart.visible = False

    if (self.method_dropdown.selected_value == None):
      self.error_message.text = "Error: No method selected. Select a method to optimize."
      self.error_message.foreground = "#ff0000"
      self.error_message.visible = True
      self.optimize_button.visible = True
      return

    self.response.visible = True
    self.response.text = 'Running optimize portfolio as background task...'
    task = anvil.server.call('launch_optimization_task', self.method_dropdown.selected_value)

    #look into task.get_termination_status()
    while (task.is_running()):
      self.response.visible = True
      dict = task.get_state('status')
      self.response.text = dict.get('status')

    # if(task.get_termination_status() == 'failed'):
    #   print(task.get_error())

    #self.response.text, self.expected_returns.text, self.volatility.text, self.sim.text = task.get_return_value()
    self.response.text, self.expected_returns.text, self.volatility.text = task.get_return_value()
    self.repeating_panel_weights.items = anvil.server.call('get_weights')
    self.efficient_frontier_plot.source = anvil.server.call('plot_efficient_frontier')
    self.pie_chart.figure = anvil.server.call('plot_pie_chart')

    self.optimize_button.visible = True
    self.returns_label.visible = True
    self.expected_returns.visible = True
    self.volatility_label.visible = True
    self.volatility.visible = True
    #self.sim_label.visible = True
    #self.sim.visible = True
    self.optimal_weights_grid.visible = True
    self.efficient_frontier_plot.visible = True
    self.pie_chart.visible = True













    
    






    
    


