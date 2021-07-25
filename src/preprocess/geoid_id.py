from shapely.geometry import Point
from shapely.ops import unary_union


def sel_geoid(strin, start, end):
    return strin[start:end]


class geoid_id(object):
    def __init__(self, df):
        # df_c0: leave original 12-digit geoid intact
        self.df_c0 = df.copy()
        self.df_c0.set_index('GEOID10', inplace=True)

        # df_c1: truncate geoid to 7 digits
        self.df_c1 = df.copy()
        self.df_c1
        self.df_c1['GEOID10'] = self.df_c1['GEOID10'].apply(str)
        self.df_c1['GEOID10'] = self.df_c1['GEOID10'].apply(sel_geoid,start=0, end=7)
        self.df_c1_shape = self.df_c1[['GEOID10', 'geometry']].groupby('GEOID10').aggregate(unary_union)
        self.df_c1_shape = self.df_c1_shape.reset_index()
        self.df_c1_shape.set_index('GEOID10', inplace=True)

        # df_c2: truncate geoid to 8 digits
        self.df_c2 = df.copy()
        self.df_c2['GEOID10'] = self.df_c2['GEOID10'].apply(str)
        self.df_c2['GEOID10'] = self.df_c2['GEOID10'].apply(sel_geoid,start=0, end=8)
        self.df_c2_shape = self.df_c2[['GEOID10', 'geometry']].groupby('GEOID10').aggregate(unary_union)
        self.df_c2_shape = self.df_c2_shape.reset_index()
        self.df_c2_shape.set_index('GEOID10', inplace=True)

        # df_c3: truncate geoid to 9 digits
        self.df_c3 = df.copy()
        self.df_c3['GEOID10'] = self.df_c3['GEOID10'].apply(str)
        self.df_c3['GEOID10'] = self.df_c3['GEOID10'].apply(sel_geoid,start=0, end=9)
        self.df_c3_shape = self.df_c3[['GEOID10', 'geometry']].groupby('GEOID10').aggregate(unary_union)
        self.df_c3_shape = self.df_c3_shape.reset_index()
        self.df_c3_shape.set_index('GEOID10', inplace=True)
    
    
    def geoid_included(self, coord):
        """Find geoid cooresponding to coord."""

        geoid = ''
        p = Point(coord)
        failed = True
        for j in range(len(self.df_c1_shape)):
            if p.within(self.df_c1_shape.iloc[j]['geometry']):
                geoid = self.df_c1_shape.index[j]
                failed = False
                break
        if failed:
            return geoid
        self.df_c2_shape_c = self.df_c2_shape.filter(like=geoid, axis=0)
        failed = True
        for j in range(len(self.df_c2_shape_c)):
            if p.within(self.df_c2_shape_c.iloc[j]['geometry']):
                geoid = self.df_c2_shape_c.index[j]
                failed = False
                break
        if failed:
            return ''
        self.df_c3_shape_c = self.df_c3_shape.filter(like=geoid, axis=0)
        failed = True
        for j in range(len(self.df_c3_shape_c)):
            if p.within(self.df_c3_shape_c.iloc[j]['geometry']):
                geoid = self.df_c3_shape_c.index[j]
                failed = False
                break
        if failed:
            return ''
        self.df_c0_c = self.df_c0.filter(like=geoid, axis=0)
        failed = True
        for j in range(len(self.df_c0_c)):
            if p.within(self.df_c0_c.iloc[j]['geometry']):
                geoid = self.df_c0_c.index[j]
                failed = False
                break
        if failed:
            return ''
        return geoid


    def find_geoid(self, coord_linestring):
        """Match points of segment to census block groups."""

        x,y = list(coord_linestring.xy)
        re = {}
        for i in range(len(x)):
            coord = [float(x[i]), float(y[i])]
            # print(coord)
            rei = self.geoid_included(coord = coord)
            if re.get(rei) == None:
                re[rei] = 1
            else:
                re[rei] += 1
        return re