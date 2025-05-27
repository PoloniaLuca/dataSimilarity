import os
import subprocess
from pathlib import Path

## Development server requires we first copy all common libraries
if __name__ == "__main__":
    # subprocess.check_output('cp -nr ../../common/ .')
    f_path = Path(__file__)
    # source = os.path.join(f_path.parent.parent.parent, 'common/')
    # dest = str(f_path.parent) + '/'

    # output =subprocess.check_output(['rsync', '-a', '-v', source, dest])
    # print(output)

from utils.constants import constants


from flask import Flask, render_template, request, flash
#import representativeness as rp
import correspondence as cr
import pandas as pd
#import numpy as np
from sklearn.decomposition import PCA
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
import os
import seaborn as sns; sns.set()
from matplotlib.backends.backend_agg import FigureCanvasAgg
#from matplotlib.figure import Figure
import io
import base64
from pathlib import Path
import subprocess
#from PIL import Image
#from io import StringIO
#import re
#import SeabornFig2Grid as sfg
#import matplotlib.gridspec as gridspec
#from sklearn import preprocessing
#import plotly.express as px
import matplotlib.patches as mpatches
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
#from numpy import array
#import cgi

UPLOAD_FOLDER = '/tmp/'
ALLOWED_EXTENSIONS = {'csv'}


app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['ALLOWED_EXTENSIONS'] = ALLOWED_EXTENSIONS



app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'


@app.context_processor
def inject_constants():
    return dict(constants=constants)

def plot_png(number, feature1, feature2, set1, set2, column_target):
    fig = create_figure(number, feature1, feature2, set1, set2, column_target)
    output = io.BytesIO()
    output.seek(0)
    FigureCanvasAgg(fig).print_png(output)
    figdata_png = base64.b64encode(output.getvalue())
    return figdata_png

def create_figure(number, feature1, feature2, set1, set2, column_target):

    if number == 1:

        sns.set(rc={'axes.facecolor':'#f5f5f5', 'figure.facecolor':'#f5f5f5', 'grid.color':'#595959', 'axes.edgecolor':'#595959'})
        #sns.set_style('whitegrid')
        fig = plt.figure()
        g=sns.JointGrid()

        scatter = sns.scatterplot(data=set1, x=feature1, y=feature2, s=100, ax=g.ax_joint, linewidth=1.5, alpha=0.35, edgecolor='none')
        sns.scatterplot(data=set2, x=feature1, y=feature2, s=100, ax=g.ax_joint, linewidth=1.5, alpha=0.35, edgecolor='none')
        scatter.set_ylabel(scatter.get_ylabel(), rotation = -90, labelpad=15)

        sns.kdeplot(data=set1, fill=True, legend=False, x=feature1, linewidth=2, ax=g.ax_marg_x)
        sns.kdeplot(data=set1, fill=True, legend=False, y=feature2, linewidth=2, ax=g.ax_marg_y)

        sns.kdeplot(data=set2, fill=True, legend=False, x=feature1, linewidth=2, ax=g.ax_marg_x)
        sns.kdeplot(data=set2, fill=True, legend=False, y=feature2, linewidth=2, ax=g.ax_marg_y)

        g.ax_marg_y.tick_params(labelright=True)
        g.ax_marg_y.grid(True, axis='x', ls=':')

        g.ax_marg_x.tick_params(labeltop=True)
        g.ax_marg_x.grid(True, axis='y', ls=':')

        plt.tight_layout()
        fig = g.fig

    else:

        if column_target != '':
            set1.drop([column_target], axis=1)
            set2.drop([column_target], axis=1)

        pca = PCA(n_components=1)

        principal_components_set1 = pca.fit_transform(set1)
        principal_components_set2 = pca.fit_transform(set2)

        principal_components_DF_set1 = pd.DataFrame(principal_components_set1, columns = ['component 1'])
        principal_components_DF_set2 = pd.DataFrame(principal_components_set2, columns = ['component 1'])


        principal_components_DF_set1['set'] = '1'
        principal_components_DF_set2['set'] = '2'

        principal_components_DF = pd.merge(principal_components_DF_set1, principal_components_DF_set2, how = 'outer')

        fig = plt.figure()

        ax1 = fig.add_subplot(2,1,1)
        sns.kdeplot(data=principal_components_DF, x="component 1", hue="set", legend=False, common_norm=False, fill=True, ax=ax1)
        ax1.set(xlabel=None)

        ax2 = fig.add_subplot(2,1,2, sharex=ax1)
        sns.kdeplot(data=principal_components_DF, x="component 1", hue="set", legend=False, common_norm=False, cumulative=True, fill=True, common_grid=True, ax=ax2)

        plt.setp(ax1.get_xticklabels(), visible=True)
        plt.gca().invert_yaxis()

        #blue_patch = mpatches.Patch(color='#1f77b4')
        #orange_patch = mpatches.Patch(color='#ff7f0e')

        #fake_handles = [blue_patch, orange_patch]

        #label = [dataset_name_set1, dataset_name_set2]

        #ax1.legend(fake_handles, label, title = 'Dataset', loc = 'center left', bbox_to_anchor=(1, 0.5))
        #ax1.legend(fake_handles, label, title = 'Dataset', bbox_to_anchor=(0.2,1.0), bbox_transform=plt.gcf().transFigure)

        fig.tight_layout()
    return fig

def standardize(set, arr, column_target):
    if column_target in arr:
        arr.remove(column_target)
    df1 = set[arr]

    std_scaler = StandardScaler()
    x_scaled = std_scaler.fit_transform(df1.values)
    df_standardize = pd.DataFrame(x_scaled, columns = df1.columns, index = df1.index)

    set.drop(labels=arr, axis="columns", inplace=True)
    set[arr] = df_standardize[arr]
    return set


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/input", methods=["POST", "GET"])
def input(test=False):
    if request.method == 'POST' or test is True:
        requestEmpty = True

        while requestEmpty:
            for file in request.files:
                if request.files[file].filename != '':
                    requestEmpty = False
                    break
            break

        if not requestEmpty or test is True:
            if test is not True:
                set1 = request.files['set1']
                set2 = request.files['set2']

                set1.save(os.path.join(app.config['UPLOAD_FOLDER'], "set1.csv"))
                set2.save(os.path.join(app.config['UPLOAD_FOLDER'], "set2.csv"))


                dataset_name_set1 = set1.filename
                dataset_name_set2 = set2.filename

                feature1 = request.form['examplefirstname']
                feature2 = request.form['examplesecondname']

                columns_to_standardize = request.form['textarea1']
                column_target = request.form['examplenamecoloumntarget']
                degree_to_visualize = request.form['degree_to_visualize']


                set1 = pd.read_csv("/tmp/set1.csv")
                set2 = pd.read_csv("/tmp/set2.csv")


            else:
                set1 = pd.read_csv("./static/testfiles/dataset1.csv", index_col=0)
                set2 = pd.read_csv("./static/testfiles/dataset2.csv", index_col=0)
                dataset_name_set1 = 'set1.csv'
                dataset_name_set2 = 'set2.csv'
                columns_to_standardize = 'Age,HCT,HGB,MCH,MCHC,MCV,RBC,WBC,PLT1,NE,LY,MO,EO,BA,NET,LYT,MOT,EOT,BAT'
                feature1 = 'LY'
                feature2 = 'PLT1'
                degree_to_visualize = '0.5'
                column_target = ''

            if degree_to_visualize != '':
                degree_to_visualize = float(degree_to_visualize)
                pvalue_percent = degree_to_visualize*100
                pvalueFormat = "{:.3f}".format(degree_to_visualize)[1:]
            else:
                pvalue = cr.degree_correspondance(set1.to_numpy(), set2.to_numpy())
                if pvalue<1:
                    pvalueFormat = "{:.3f}".format(pvalue)[1:]
                else:
                    pvalueFormat = "{:.3f}".format(pvalue)
                pvalue_percent = pvalue*100

            if columns_to_standardize != '':
                arr = columns_to_standardize.split(',')
                set1 = standardize(set1, arr, column_target)
                set2 = standardize(set2, arr, column_target)

            if column_target != '':
                principal_feature_DF = pd.merge(set1, set2, how = 'outer')
                y_clf=principal_feature_DF[column_target]
                X_clf=principal_feature_DF.drop([column_target], axis=1)
                select = SelectKBest(score_func=mutual_info_classif, k=2)
                z = select.fit_transform(X_clf, y_clf)
                mask = select.get_support()
                new_features = X_clf.columns[mask]
                feature1 = new_features[0]
                feature2 = new_features[1]

            fig_dataset1a = plot_png(1, feature1, feature2, set1, set2, column_target)
            fig_dataset1b = plot_png(2, feature1, feature2, set1, set2, column_target)
            return render_template("results.html", pvalue=pvalueFormat, pvalue_percent=pvalue_percent, plot_dataset1a=fig_dataset1a.decode('utf8'), plot_dataset1b=fig_dataset1b.decode('utf8'), dataset_name_set1 = dataset_name_set1, dataset_name_set2 = dataset_name_set2)
        else:
            flash("No file uploaded")
    return render_template("input.html")

@app.route('/')
def home():
    return render_template('home.html')

@app.route("/test/", methods=["POST", "GET"])
def test():
    return input(True)

@app.route("/deploy")
def deploy():
    output = subprocess.check_output('./lib/deploy.sh', stderr=subprocess.STDOUT).replace(b'\n', b'<br />')

    return output

if __name__ == "__main__":
    app.run(debug=True, port=5002)
