import numpy as np
import pandas as pd
from itertools import product
from more_itertools import windowed

# baseado na abordagem de
# https://tylermarrs.com/posts/deriving-markov-transition-matrices/

class Transition():
  '''
  Isso aqui é um objeto Transition, que parametriza uma transição.
  '''
  def __init__(self, current_state, next_state, **kwargs):
    self.current_state = current_state
    self.next_state = next_state
    self.freq = kwargs.pop('freq', 0.0)
    self.prob = 0.0
    # basicamente o índice em state space
    # vou mudar isso aqui
    self.transition_metadata = kwargs.pop('transition_metadata', {})
    # delta é um modificador de freq
    # é uma variável temporária que serve pra que a gente altere
    # momentaneamente a frequência de um evento qualquer, sem que 
    # se modifique self.freq de cara
    self.delta = 0
      
  def increment_freq(self, x=1):
    # essa é a frequencia REAL da transição
    # uma vez que temos todos os dados e podemos filtrar depois,
    # esse valor aqui é imutável!! Já o valor de TransitionMatrix.freq_mat é...
    self.freq += x
  
  def set_delta(self, x, apply=False):
    assert x >= 0, "Não existe quantidade negativa... update é a partir de valor final, não de delta"
    self.delta = (x - self.freq) if x > 0 else 0 
    # se mandar aplicar, então muda, mas mantém o histórico da alteração
    # pra ser reversível
    self.freq += self.delta if apply else 0

  def update_metadata(self, metadata, override=False):
    # atualiza os metadados de uma transição
    # melhor maneira de adicionar tags e bgls pra segmentação
    self.transition_metadata = {} if override else self.transition_metadata
    for key, val in metadata.iteritems():
        if isinstance(val, collections.Mapping):
            tmp = update(self.transition_metadata.get(key, { }), val)
            self.transition_metadata[key] = tmp
        elif isinstance(val, list):
            self.transition_metadata[key] = (self.transition_metadata.get(key, []) + val)
        else:
            self.transition_metadata[key] = metadata[key]
  
  def __str__(self):
    # representação em string
    return f'{self.current_state} --> {self.next_state}'
  
  def __repr__(self):
    # quando retornar o objeto, é isso o que aparece...
    # é só representação...
    return self.__str__()

  def __eq__(self, other):
        if not isinstance(other, Transition):
            return False
        
        return self.current_state == other.current_state \
            and self.next_state == other.next_state

class TransitionMatrix():
  '''
  Isso aqui é a engine de cálculo das cadeias de markov
  '''
  def __init__(self, state_space):
    self.generate_state_space(state_space)
    self.__initialize_transitions()
    # state_freqs não é ideal criar aqui
    # uma vez que isso erá derivado da freq_mat
    # e indexado diretamente para os índices dos estados
  
  def generate_state_space(self, unique_states):
    # faz um dict com todos os estados enumerados
    self.state_space = {state:i for i,state in enumerate(unique_states)}
  
  def update_transitions(self, data, **kwargs):
    '''
    Como funciona o update da TransitionMatrix:
    Existem dois modos de atualizar as frequências:
    1- Externo, por um dicionário contendo whatever quantos pares
    {(i,j): {count, metadata: {}}} forem.
    2- Interno: por meio de freq_mat. 
    O segundo modo é utilizado, pois a engine de cálculo de probabilidade ou de
    simulação recebe np.array e não um objeto Transition. Então, caso a gente
    calcule qualquer coisa complexa, esse cálculo é realizado em freq_mat
    e depois passado pra Transitions. Caso a gente altere Transitions, então
    o update é repassado pra freq_mat. Por isso que no update externo, a gente
    dá flush em freq_mat
    '''
    def external_update(data):
      # atualiza as 
      # data é um dict com [(ev1,ev2): {'count': count,'metadata':{}}}
      aux = lambda x: (self.state_space[x[0]],self.state_space[x[1]])
      for d in data:
        if d[1]['count'] != 0:
          # bom... se não tem nada pra updatar... não updata!! =D
          self.transitions[aux(d[0])].increment_freq(d[1]['count'])
        if d[1]['metadata'] is not None:
          self.transitions[aux(d[0])].update_metadata(d[1][1])
      self.__update_freq_mat()
    def internal_update(data):
      # esse kwargs aqui é lá de cima... acho que isso é uma péssima prática.
      apply = kwargs.pop('apply', False)
      for idx in self.transitions.keys():
        self.transitions[idx].set_delta(self.freq_mat[idx], apply)
    
    if isinstance(data, (list,tuple)):
      external_update(data)
    elif isinstance(data, (np.ndarray, np.generic)):
      internal_update(data)
    self.__compute_states_freq()
    self.__compute_probabilities()

  def __update_freq_mat(self):
    # atualiza self.freq_mat baseado em self.transitions (que é atualizada primeiro)
    # TODAS as UPDATES são realizadas PRIMEIRO em self.transitions!!!!!
    # logo, tudo quanto que é update finalizado, tem que ser IGUAL a 
    # Transitions(n,m).freq
    # então toda vez que dá update em 
    for transition in self.transitions.items():
      self.freq_mat[transition[0]] = transition[1].freq

  def __initialize_transitions(self):
    # inicializa a lista de transições
    # precisa de um state_space
    # data virá de um parser em outro objeto
    # recebe um dict com [(k_i,k_j): count], onde v é 
    trans_map = list(product(self.state_space.keys(),repeat=2))
    # one-liners pra melhorar a leitura
    aux = lambda x: (self.state_space[x[0]],self.state_space[x[1]])
    create_transition = lambda x: Transition(x[0],x[1])
    self.transitions = {aux(trans): create_transition(trans) \
                        for trans in trans_map}
    # inicializa junto a freq_mat
    self.freq_mat = np.zeros([len(self.state_space)]*2)

  def __compute_probabilities(self):
    # método para computar as probs.
    # nada mais é do que freq_mat/freq_mat.sum(ax1)
    self.transition_probs = self.freq_mat / self.current_states_freq
    # adicionar algo caso dê nan... testar!

  def __compute_states_freq(self):
    # método pra calcular a freq de cada estado
    # freq.sum(ax1) 
    # mas precisa do método pra fazer o update nos objetos tbm.
    self.current_states_freq = self.freq_mat.sum(axis=1)

  def save_transition_data(self, pth):
    # esse método salva a lista de transitions 
    # com os metadados. Depois é só carregar e calcular 
    # a matriz, quando inicializar os negócios.
    raise NotImplemented('Também não fiz isso aqui ainda.')

  def load_transition_data(self, pth):
    # método pra carregar as transições
    raise NotImplemented('Não fiz')
    
class MCParser():
  '''
   Isso aqui serve pra fazer parsing num dataframe de eventos.
  '''
  @staticmethod
  def prep_event_dataset(event_data, apply_end=True):
    states = list(event_data.event_name.unique())
    event_data = event_data.copy().groupby(['user_pseudo_id'])['event_name'].apply(list).reset_index(name='paths') 
    # adding end event, just to check some stuff
    if apply_end:
      _ = event_data.paths.apply(lambda x: x.append('end'))
    #self.event_data = event_data
    #self.states = states
    return event_data, states

  @staticmethod
  def calc_transitions(paths_df, state_space):
    # pega o paths df, calcula as frequências e adiciona os users
    # nos metadata
    # dict temporário, onde vamos guardar as contagens
    # eu poderia concatenar tudo numa parada só.
    # Mas eu quero modularizar o MC e aplicar pra outros tipos de dados
    # então é melhor que fique aberto pra qualquer tipo de dado
    transition_data = {idx: {'counts': 0,
                              'metadata': {'users': []}}
                      for idx in list(product(state_space,repeat=2))}
    for idx, row in a.iterrows():
      for pair in list(more_itertools.windowed(row['paths'],2)):
        transition_data[pair]['counts'] += 1
        transition_data[pair]['metadata']['users'].append(row['user_pseudo_id'])

    return transition_data 