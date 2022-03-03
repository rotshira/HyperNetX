from hypernetx import *
import pandas as pd
from pandas.api.types import CategoricalDtype
import numpy as np
from collections import defaultdict, OrderedDict, UserList
from collections.abc import Hashable
from scipy.sparse import csr_matrix


class StaticEntity(object):
    """
    A new Entity object using pandas.DataFrame as the base data structure

    TODO: split data parameter into mulitple parameters - one for raw data and one for
          sparse tensor indices
    TODO: allow addition of rows of data from dict of lists or lists of lists
    TODO: allow removal of rows of data

    Parameters
    ----------
    data : pandas.DataFrame, dict of lists, list of lists, or 2d numpy.ndarray
        If a pandas.DataFrame, representation of entity data with 2 or more dimensions,
        If a dict of lists or list of lists, representation of entity data with exactly
        2 dimensions,
        If a 2d numpy.ndarray of ints, representation of entity data with 2 or more
        dimensions (as sparse tensor indices for incidence tensor)
    static : bool, default=True
        If True, data may not be altered, and the state dict will never be cleared
        If False, rows may be added to and removed from data, and updates will clear the
        state dict
    labels : OrderedDict of lists, optional
        User defined labels corresponding to integers in data, only used when data is
        given as a numpy.ndarray of sparse tensor indices
    uid : Hashable, optional
    weights : array-like or Hashable, optional
        User specified cell weights corresponding to data,
        If array-like, length must equal number of rows in data.
        If Hashable, must be the name of a column in data.
        Otherwise, weight for all rows is assumed to be 1.
    aggregateby : str, optional, default='sum'
        Method to aggregate cell weights of duplicate rows of data.
        If None, duplicate rows will be dropped without aggregating cell weights

    Attributes
    ----------
    uid: Hashable
    static: bool
    dataframe: pandas.DataFrame
    data_cols: list of Hashables
    cell_weight_col: Hashable
    dimsize: int
    state_dict: dict
        The state_dict holds all attributes that must be recomputed when the data is
        updated. The values for these attributes will be computed as needed if they do
        not already exist in the state_dict.

        data : numpy.ndarray
            sparse tensor indices for incidence tensor
        labels: dict of lists
            labels corresponding to integers in sparse tensor indices given by data
        cell_weights: dict
            keys are rows of labeled data as tuples, values are cell weight of the row
        dimensions: tuple of ints
            tuple of number of unique labels in each column/dimension of the data
        uidset: dict of sets
            keys are columns of the dataframe, values are the set of unique labeled
            values in the column
        elements: defaultdict of nested dicts
            top level keys are column names, elements[col1][col2] holds the elements
            by level dict with col1 as level 1 and col2 as level 2 (keys are unique
            values of col1, values are list of unique values of col2 that appear in a
            row of data with the col1 value key)

    """

    def __init__(
        self,
        data=None,
        static=True,
        labels=None,
        uid=None,
        weights=None,
        aggregateby="sum",
    ):
        # set unique identifier
        self._uid = uid
        self._properties = {}

        # if static, the original data cannot be altered
        # the state dict stores all computed values that may need to be updated if the
        # data is altered - the dict will be cleared when data is added or removed
        self._static = static
        self._state_dict = {}

        # raw entity data is stored in a DataFrame for basic access without the need for
        # any label encoding lookups
        if isinstance(data, pd.DataFrame):
            self._dataframe = data.copy()

        # if the raw data is passed as a dict of lists or a list of lists, we convert it
        # to a 2-column dataframe by exploding each list to cover one row per element
        # for a dict of lists, the first level/column will be filled in with dict keys
        # for a list of N lists, 0,1,...,N will be used to fill the first level/column
        elif isinstance(data, (dict, list)):
            # convert dict of lists to 2-column dataframe
            data = pd.Series(data).explode()
            self._dataframe = pd.DataFrame({0: data.index, 1: data.values})

        # if a 2d numpy ndarray is passed, store it as both a DataFrame and an ndarray
        # in the state dict
        elif isinstance(data, np.ndarray) and data.ndim == 2:
            self._state_dict["data"] = data
            self._dataframe = pd.DataFrame(data)
            # if a dict of labels was passed, use keys as column names in the DataFrame,
            # translate the dataframe, and store the dict of labels in the state dict
            if isinstance(labels, dict) and len(labels) == len(self._dataframe.columns):
                self._dataframe.columns = labels.keys()
                self._state_dict["labels"] = labels

                for col in self._dataframe:
                    self._dataframe[col] = pd.Categorical.from_codes(
                        self._dataframe[col], categories=labels[col]
                    )

        # create an empty Entity
        # TODO: clean this up to be less hacky?
        else:
            self._dataframe = pd.DataFrame()
            self._data_cols = []
            self._cell_weight_col = None
            self._dimsize = 0
            return

        # assign a new or existing column of the dataframe to hold cell weights
        self._dataframe, self._cell_weight_col = assign_weights(
            self._dataframe, weights=weights
        )
        # store a list of columns that hold entity data (not properties or weights)
        self._data_cols = list(self._dataframe.columns.drop(self._cell_weight_col))

        # each entity data column represents one dimension of the data
        # (data updates can only add or remove rows, so this isn't stored in state dict)
        self._dimsize = len(self._data_cols)

        # remove duplicate rows and aggregate cell weights as needed
        self._dataframe, _ = remove_row_duplicates(
            self._dataframe,
            self._data_cols,
            weights=self._cell_weight_col,
            aggregateby=aggregateby,
        )

        # set the dtype of entity data columns to categorical (simplifies encoding, etc.)
        self._dataframe[self._data_cols] = self._dataframe[self._data_cols].astype(
            "category"
        )

    @property
    def data(self):
        if "data" not in self._state_dict:
            if self.empty:
                self._state_dict["data"] = np.zeros((0, 0), dtype=int)
            else:
                # assumes dtype of data cols is categorical and dataframe not altered
                self._state_dict["data"] = (
                    self._dataframe[self._data_cols]
                    .apply(lambda x: x.cat.codes)
                    .to_numpy()
                )

        return self._state_dict["data"]

    @property
    def labels(self):
        if "labels" not in self._state_dict:
            if self.empty:
                self._state_dict["labels"] = {}
            else:
                # assumes dtype of data cols is categorical and dataframe not altered
                self._state_dict["labels"] = (
                    self._dataframe[self._data_cols]
                    .apply(lambda x: x.cat.categories.to_list())
                    .to_dict()
                )

        return self._state_dict["labels"]

    @property
    def cell_weights(self):
        if "cell_weights" not in self._state_dict:
            if self.empty:
                self._state_dict["cell_weights"] = {}
            else:
                self._state_dict["cell_weights"] = self._dataframe.set_index(
                    self._data_cols
                )[self._cell_weight_col].to_dict()

        return self._state_dict["cell_weights"]

    @property
    def dimensions(self):
        if "dimensions" not in self._state_dict:
            if self.empty:
                self._state_dict["dimensions"] = tuple()
            else:
                self._state_dict["dimensions"] = tuple(
                    self._dataframe[self._data_cols].nunique()
                )

        return self._state_dict["dimensions"]

    @property
    def dimsize(self):
        return self._dimsize

    @property
    def properties(self):
        return self._properties

    @property
    def uid(self):
        return self._uid

    @property
    def uidset(self):
        return self.uidset_by_level(0)

    @property
    def children(self):
        return self.uidset_by_level(1)

    def uidset_by_level(self, level):
        col = self._data_cols[level]
        return self.uidset_by_column(col)

    def uidset_by_column(self, column):
        if "uidset" not in self._state_dict:
            self._state_dict["uidset"] = {}
        if column not in self._state_dict["uidset"]:
            self._state_dict["uidset"][column] = set(
                self._dataframe[column].dropna().unique()
            )

        return self._state_dict["uidset"][column]

    @property
    def elements(self):
        return self.elements_by_level(0, 1)

    @property
    def incidence_dict(self):
        return self.elements

    @property
    def memberships(self):
        return self.elements_by_level(1, 0)

    def elements_by_level(self, level1, level2):
        col1 = self._data_cols[level1]
        col2 = self._data_cols[level2]
        return self.elements_by_column(col1, col2)

    def elements_by_column(self, col1, col2):
        if "elements" not in self._state_dict:
            self._state_dict["elements"] = defaultdict(dict)
        if col2 not in self._state_dict["elements"][col1]:
            elements = self._dataframe.groupby(col1)[col2].unique()
            self._state_dict["elements"][col1][col2] = elements.apply(
                UserList
            ).to_dict()

        return self._state_dict["elements"][col1][col2]

    @property
    def dataframe(self):
        return self._dataframe

    @property
    def isstatic(self):
        return self._static

    def size(self, level=0):
        return self.dimensions[level]

    @property
    def empty(self):
        return self._dimsize == 0

    def is_empty(self, level=0):
        return self.empty or self.size(level) == 0

    def __len__(self):
        return self.dimensions[0]

    def __contains__(self, item):
        return item in np.concatenate(list(self.labels.values()))

    def __getitem__(self, item):
        return self.elements[item]

    def __iter__(self):
        return iter(self.elements)

    def __call__(self, label_index=0):
        return iter(self.labels[self._data_cols[label_index]])

    def __repr__(self):
        return (
            self.__class__.__name__
            + f"({self._uid},{list(self.uidset)},{self.properties})"
        )

    def index(self, column, value=None):
        if "keyindex" not in self._state_dict:
            self._state_dict["keyindex"] = {}
        if column not in self._state_dict["keyindex"]:
            self._state_dict["keyindex"][column] = self._dataframe[
                self._data_cols
            ].columns.get_loc(column)

        if value is None:
            return self._state_dict["keyindex"][column]

        if "index" not in self._state_dict:
            self._state_dict["index"] = defaultdict(dict)
        if value not in self._state_dict["index"][column]:
            self._state_dict["index"][column][value] = self._dataframe[
                column
            ].cat.categories.get_loc(value)

        return (
            self._state_dict["keyindex"][column],
            self._state_dict["index"][column][value],
        )

    def indices(self, column, values):
        if isinstance(values, Hashable):
            values = [values]

        if "index" not in self._state_dict:
            self._state_dict["index"] = defaultdict(dict)
        for v in values:
            if v not in self._state_dict["index"][column]:
                self._state_dict["index"][column][v] = self._dataframe[
                    column
                ].cat.categories.get_loc(v)

        return [self._state_dict["index"][column][v] for v in values]

    def translate(self, level, index):
        column = self._data_cols[level]

        if isinstance(index, int):
            return self.labels[column][index]
        
        return [self.labels[column][i] for i in index]

    def translate_arr(self, coords):
        assert len(coords) == self._dimsize

        translation = []
        for level, index in enumerate(coords):
            translation.append(self.translate(level, index))

        return translation

    def level(self, item, min_level=0, max_level=None, return_index=True):
        if max_level is None or max_level >= self._dimsize:
            max_level = self._dimsize - 1

        columns = self._data_cols[min_level : max_level + 1]
        levels = range(min_level, max_level + 1)

        for col, lev in zip(columns, levels):
            if item in self.labels[col]:
                if return_index:
                    return self.index(col, item)

                return lev

        print(f'"{item}" not found.')
        return None

    def add(self, data, aggregateby="sum"):
        # TODO: add from other data types
        if self.isstatic:
            raise HyperNetXError("Cannot add data to a static Entity")
        if isinstance(data, pd.DataFrame) and all(
            col in data for col in self._data_cols
        ):
            new_data = pd.concat((self._dataframe, data), ignore_index=True)
            new_data[self._cell_weight_col] = new_data[self._cell_weight_col].fillna(1)

            self._dataframe, _ = remove_row_duplicates(
                new_data,
                self._data_cols,
                weights=self._cell_weight_col,
                aggregateby=aggregateby,
            )

            self._dataframe[self._data_cols] = self._dataframe[self._data_cols].astype(
                "category"
            )
            # TODO: check to see if we really need to clear everything
            self._state_dict.clear()
    def encode(self, data):
        encoded_array = data.apply(lambda x: x.cat.codes).to_numpy()
        return encoded_array

    def incidence_matrix(self, level1=0, level2=1, weights=False, aggregateby=None, index=False):
        if self.dimsize < 2:
            warnings.warn("Incidence matrix requires two levels of data.")
            return None

        data_cols = [self._data_cols[level1], self._data_cols[level2]]
        result = ''
        if weights:
            if self.dimsize > 2:
                temp, temp_weights = remove_row_duplicates(self.dataframe, data_cols, weights=self._cell_weight_col, aggregateby=aggregateby)
            else:
                temp, temp_weights = self.data[[data_cols]], self.cell_weights
            print(temp_weights)

            # if isinstance(weights, dict):
            #     cat1 = self.keys[level1]
            #     cat2 = self.keys[level2]
            #     for k, v in weights:
            #         try:
            #             tdx = (self.index(cat1, k[0]), self.index(cat2, k[1]))
            #         except:
            #             HyperNetXError(
            #                 f"{k} is not recognized as belonging to this system."
            #             )
            #         if temp_weights[tdx] != 0:
            #             temp_weights[tdx] = v
            temp_weights = [temp_weights[tuple(t)] for t in temp]
            dtype = int if aggregateby == "count" else float
            result = csr_matrix(
                (temp_weights, temp.transpose()), dtype=dtype
            ).transpose()
        else:
            if self.dimsize > 2:
                temp, _ = remove_row_duplicates(self.dataframe, data_cols)
            else:
                temp = self.dataframe[data_cols] 

            temp = temp[data_cols]
            temp = temp.astype("category")
            temp = self.encode(temp)
            result = csr_matrix((np.ones(len(temp)), (temp[:,1], temp[:,0])), dtype=int)

        return result

    def restrict_to_levels(self, levels, weights=False, aggregateby="sum", uid=None):
        if levels[0] >= self.dimsize:
            return self.__class__()
        else:
            if weights:
                weights = self._cell_weight_col
            else: 
                weights = None
            cols = [self._data_cols[level] for level in levels]
            newDataframe = self._dataframe[cols]
            return StaticEntity(
                    data=newDataframe,
                    weights=weights,
                    aggregateby=aggregateby,
                )


class StaticEntitySet(StaticEntity):
    def __init__(
        self,
        entity=None,
        data=None,
        static=True,
        labels=None,
        uid=None,
        level1=0,
        level2=1,
        weights=None,
        keep_weights=True,
        aggregateby="sum",
    ):

        if isinstance(entity, StaticEntity):
            if keep_weights:
                weights = entity._cell_weight_col
            data = entity.dataframe

        if isinstance(data, pd.DataFrame):
            if isinstance(weights, Hashable) and weights in data:
                columns = data.columns.drop(weights)[[level1, level2]]
                columns = columns.append(pd.Index([weights]))
            else:
                columns = data.columns[[level1, level2]]

            data = data[columns]

        elif isinstance(data, np.ndarray) and data.ndim == 2:
            data = data[:, (level1, level2)]

        if isinstance(labels, dict):
            label_keys = list(labels)
            columns = (label_keys[level1], label_keys[level2])
            labels = {col: labels[col] for col in columns}

        super().__init__(
            data=data,
            static=static,
            labels=labels,
            uid=uid,
            weights=weights,
            aggregateby=aggregateby,
        )


def assign_weights(df, weights=None, weight_col="cell_weights"):
    """
    Parameters
    ----------
    df : pandas.DataFrame
        A DataFrame to assign a weight column to
    weights : array-like or Hashable, optional
        If numpy.ndarray with the same length as df, create a new weight column with
        these values.
        If Hashable, must be the name of a column of df to assign as the weight column
        Otherwise, create a new weight column assigning a weight of 1 to every row
    weight_col : Hashable
        Name for new column if one is created (not used if the name of an existing
        column is passed as weights)

    Returns
    -------
    df : pandas.DataFrame
        The original DataFrame with a new column added if needed
    weight_col : str
        Name of the column assigned to hold weights
    """
    if isinstance(weights, (list, np.ndarray)) and len(weights) == len(df):
        df[weight_col] = weights
    elif isinstance(weights, Hashable) and weights in df:
        weight_col = weights
    else:
        df[weight_col] = np.ones(len(df), dtype=int)

    return df, weight_col


def remove_row_duplicates(df, data_cols, weights=None, aggregateby="sum"):
    """
    Removes and aggregates duplicate rows of a DataFrame using groupby

    Parameters
    ----------
    df : pandas.DataFrame
        A DataFrame to remove or aggregate duplicate rows from
    data_cols : list
        A list of column names in df to perform the groupby on / remove duplicates from
    weights : array-like or Hashable, optional
        Argument passed to assign_weights
    aggregateby : str, optional, default='sum'
        A valid aggregation method for pandas groupby
        If None, drop duplicates without aggregating weights

    Returns
    -------
    df : pandas.DataFrame
        The DataFrame with duplicate rows removed or aggregated
    weight_col : Hashable
        The name of the column holding aggregated weights, or None if aggregateby=None
    """
    df = df.copy()
    categories = {}
    for col in data_cols:
        if df[col].dtype.name == "category":
            categories[col] = df[col].cat.categories
            df[col] = df[col].astype(categories[col].dtype)

    if not aggregateby:
        df = df.drop_duplicates(subset=data_cols)

    df, weight_col = assign_weights(df, weights=weights)

    if aggregateby:
        df = df.groupby(data_cols, as_index=False, sort=False).agg(
            {weight_col: aggregateby}
        )

    for col in categories:
        df[col] = df[col].astype(CategoricalDtype(categories=categories[col]))

    return df, weight_col
