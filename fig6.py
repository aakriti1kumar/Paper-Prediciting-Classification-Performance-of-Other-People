import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import matplotlib.patches as mpatches
import set_style
import combo_data

set_style.set_style()


def open_file(filename) -> object:
    with open(f"{filename}", "rb") as handle:
        return pickle.load(handle)


# Read all data
true = open_file("true_params.pkl")
self = open_file("self_params.pkl")
# self_f = open_file("self_sim_f.pkl")
# self_nf = open_file("self_sim_nf.pkl")

hyp1_f = open_file("hyp1-feedback/hyp1_feedback64.pkl")
hyp2_f = open_file("hyp2-feedback/hyp2_feedback64.pkl")
hyp3_f = open_file("hyp3-feedback/hyp3_feedback.pkl")

hyp1_nf = open_file("hyp1-nofeedback/hyp1_nofeedback.pkl")
hyp2_nf = open_file("hyp2-nofeedback/hyp2_nofeedback.pkl")
hyp3_nf = open_file("hyp3-nofeedback/hyp3_nofeedback.pkl")
#
# New combo 1
hyp1_f_c1 = open_file("/Users/aakritikumar/Desktop/Lab/ToM-pycharm/combo1/hyp1-feedback/hyp1_feedback64.pkl")
hyp2_f_c1 = open_file("/Users/aakritikumar/Desktop/Lab/ToM-pycharm/combo1/hyp2-feedback/hyp2_feedback64.pkl")
hyp3_f_c1 = open_file("/Users/aakritikumar/Desktop/Lab/ToM-pycharm/combo1/hyp3-feedback/hyp3_feedback.pkl")

hyp1_nf_c1 = open_file("/Users/aakritikumar/Desktop/Lab/ToM-pycharm/combo1/hyp1-nofeedback/hyp1_nofeedback.pkl")
hyp2_nf_c1 = open_file("/Users/aakritikumar/Desktop/Lab/ToM-pycharm/combo1/hyp2-nofeedback/hyp2_nofeedback.pkl")
hyp3_nf_c1 = open_file("/Users/aakritikumar/Desktop/Lab/ToM-pycharm/combo1/hyp3-nofeedback/hyp3_nofeedback.pkl")

# New combo 2
hyp1_f_c2 = open_file("/Users/aakritikumar/Desktop/Lab/ToM-pycharm/combo2/hyp1-feedback/hyp1_feedback64.pkl")
hyp2_f_c2 = open_file("/Users/aakritikumar/Desktop/Lab/ToM-pycharm/combo2/hyp2-feedback/hyp2_feedback64.pkl")
hyp3_f_c2 = open_file("/Users/aakritikumar/Desktop/Lab/ToM-pycharm/combo2/hyp3-feedback/hyp3_feedback.pkl")

hyp1_nf_c2 = open_file("/Users/aakritikumar/Desktop/Lab/ToM-pycharm/combo2/hyp1-nofeedback/hyp1_nofeedback.pkl")
hyp2_nf_c2 = open_file("/Users/aakritikumar/Desktop/Lab/ToM-pycharm/combo2/hyp2-nofeedback/hyp2_nofeedback.pkl")
hyp3_nf_c2 = open_file("/Users/aakritikumar/Desktop/Lab/ToM-pycharm/combo2/hyp3-nofeedback/hyp3_nofeedback.pkl")

# New combo 3
hyp1_f_c3 = open_file("/Users/aakritikumar/Desktop/Lab/ToM-pycharm/combo3/hyp1-feedback/hyp1_feedback64.pkl")
hyp2_f_c3 = open_file("/Users/aakritikumar/Desktop/Lab/ToM-pycharm/combo3/hyp2-feedback/hyp2_feedback64.pkl")
hyp3_f_c3 = open_file("/Users/aakritikumar/Desktop/Lab/ToM-pycharm/combo3/hyp3-feedback/hyp3_feedback.pkl")

hyp1_nf_c3 = open_file("/Users/aakritikumar/Desktop/Lab/ToM-pycharm/combo3/hyp1-nofeedback/hyp1_nofeedback.pkl")
hyp2_nf_c3 = open_file("/Users/aakritikumar/Desktop/Lab/ToM-pycharm/combo3/hyp2-nofeedback/hyp2_nofeedback.pkl")
hyp3_nf_c3 = open_file("/Users/aakritikumar/Desktop/Lab/ToM-pycharm/combo3/hyp3-nofeedback/hyp3_nofeedback.pkl")

# New combo 3
hyp1_f_c4 = open_file("/Users/aakritikumar/Desktop/Lab/ToM-pycharm/combo4/hyp1-feedback/hyp1_feedback64.pkl")
hyp2_f_c4 = open_file("/Users/aakritikumar/Desktop/Lab/ToM-pycharm/combo4/hyp2-feedback/hyp2_feedback64.pkl")
hyp3_f_c4 = open_file("/Users/aakritikumar/Desktop/Lab/ToM-pycharm/combo4/hyp3-feedback/hyp3_feedback.pkl")

hyp1_nf_c4 = open_file("/Users/aakritikumar/Desktop/Lab/ToM-pycharm/combo4/hyp1-nofeedback/hyp1_nofeedback.pkl")
hyp2_nf_c4 = open_file("/Users/aakritikumar/Desktop/Lab/ToM-pycharm/combo4/hyp2-nofeedback/hyp2_nofeedback.pkl")
hyp3_nf_c4 = open_file("/Users/aakritikumar/Desktop/Lab/ToM-pycharm/combo4/hyp3-nofeedback/hyp3_nofeedback.pkl")

# for mean true scores on plots
df = pd.read_csv("Exp2_Estimation.csv")
actcorrectothermean_top = df[(df['otherworkerperf'] == 'Top') & (df.conditionshowFeedback == 1)].actCorrectOther.mean()
actcorrectothermean_bottom = df[
    (df['otherworkerperf'] == 'Bottom') & (df.conditionshowFeedback == 1)].actCorrectOther.mean()

actcorrectothermean_top_nfb = df[
    (df['otherworkerperf'] == 'Top') & (df.conditionshowFeedback == 0)].actCorrectOther.mean()
actcorrectothermean_bottom_nfb = df[
    (df['otherworkerperf'] == 'Bottom') & (df.conditionshowFeedback == 0)].actCorrectOther.mean()

# actcorrectselfmean_top = df[(df['otherworkerperf'] == 'Top') & (df.conditionshowFeedback == 1)].actCorrectSelf.mean()
# actcorrectselfmean_bottom = df[
#     (df['otherworkerperf'] == 'Bottom') & (df.conditionshowFeedback == 1)].actCorrectSelf.mean()
#
# actcorrectselfmean_top_nfb = df[
#     (df['otherworkerperf'] == 'Top') & (df.conditionshowFeedback == 0)].actCorrectSelf.mean()
# actcorrectselfmean_bottom_nfb = df[
#     (df['otherworkerperf'] == 'Bottom') & (df.conditionshowFeedback == 0)].actCorrectSelf.mean()

# labels of top and bottom performers
df = pd.read_csv("Exp2_Estimation.csv")
df = df[df.conditionshowFeedback == 1]
df = df.sort_values(['uid'], ascending=[True], kind="mergesort")
df = df.drop_duplicates(subset=['uid'])
idx_f = df.uid.unique() - 1
labels_TB = df.otherworkerperf.values

df = pd.read_csv("Exp2_Estimation.csv")
df = df[df.conditionshowFeedback == 0]
df = df.sort_values(['uid'], ascending=[True], kind="mergesort")
df = df.drop_duplicates(subset=['uid'])
idx_nf = df.uid.unique() - 1
labels_TB_nf = df.otherworkerperf.values

uid_full = (np.linspace(0, 127, 128)).astype(int)
id_nf_c1 = list(set(uid_full) & set((combo_data.combo1_other_id).astype(int)))
id_nf_c2 = list(set(uid_full) & set((combo_data.combo2_other_id).astype(int)))
id_nf_c3 = list(set(uid_full) & set((combo_data.combo3_other_id).astype(int)))
id_nf_c4 = list(set(uid_full) & set((combo_data.combo4_other_id).astype(int)))

labels_TB_nc1 = [labels_TB[int(combo_data.combo1_other_id[i])] for i in range(len(combo_data.combo1_other_id))]
labels_TB_nf_nc1 = [labels_TB_nf[int(id_nf_c1[i])] for i in range(len(id_nf_c1))]

labels_TB_nc2 = [labels_TB[int(combo_data.combo2_other_id[i])] for i in range(len(combo_data.combo2_other_id))]
labels_TB_nf_nc2 = [labels_TB_nf[int(id_nf_c2[i])] for i in range(len(id_nf_c2))]

labels_TB_nc3 = [labels_TB[int(combo_data.combo3_other_id[i])] for i in range(len(combo_data.combo3_other_id))]
labels_TB_nf_nc3 = [labels_TB_nf[int(id_nf_c3[i])] for i in range(len(id_nf_c3))]

labels_TB_nc4 = [labels_TB[int(combo_data.combo4_other_id[i])] for i in range(len(combo_data.combo4_other_id))]
labels_TB_nf_nc4 = [labels_TB_nf[int(id_nf_c4[i])] for i in range(len(id_nf_c4))]


def makedf(df, df_nc1, df_nc2, df_nc3, df_nc4, labels, labels_nc1, labels_nc2, labels_nc3, labels_nc4, idx):
    df_simdata = pd.DataFrame()
    df_simdata['User_ID'] = np.repeat(np.linspace(1, 64 * 5, 64 * 5), 16)
    blocks = np.linspace(1, 16, 16)
    df_simdata['Blocks'] = 0
    for i in range(64 * 5):
        df_simdata['Blocks'][i * 16:(i + 1) * 16] = blocks

    df_simdata = df_simdata.astype(int)

    # Add self scores
    df_simdata['SelfScore'] = 0
    df_simdata['SelfScore'][:(64 * 16)] = self['Sim_Y'][idx].flatten()
    df_simdata['SelfScore'][(64 * 16):(64 * 16 * 2)] = self['Sim_Y'][idx].flatten()
    df_simdata['SelfScore'][(64 * 16 * 2):(64 * 16 * 3)] = self['Sim_Y'][idx].flatten()
    df_simdata['SelfScore'][(64 * 16 * 3):(64 * 16 * 4)] = self['Sim_Y'][idx].flatten()
    df_simdata['SelfScore'][(64 * 16 * 4):(64 * 16 * 5)] = self['Sim_Y'][idx].flatten()

    # Add other score
    df_simdata['OtherScore'] = 0
    df_simdata['OtherScore'][:(64 * 16)] = df['Sim_OtherEst'].flatten()
    df_simdata['OtherScore'][(64 * 16):(64 * 16 * 2)] = df_nc1['Sim_OtherEst'].flatten()
    df_simdata['OtherScore'][(64 * 16 * 2):(64 * 16 * 3)] = df_nc2['Sim_OtherEst'].flatten()
    df_simdata['OtherScore'][(64 * 16 * 3):(64 * 16 * 4)] = df_nc3['Sim_OtherEst'].flatten()
    df_simdata['OtherScore'][(64 * 16 * 4):(64 * 16 * 5)] = df_nc4['Sim_OtherEst'].flatten()

    # Add top bottom label
    df_simdata['TBlabel'] = 0
    df_simdata['TBlabel'][:(64 * 16)] = np.repeat(labels, 16)
    df_simdata['TBlabel'][(64 * 16):(64 * 16 * 2)] = np.repeat(labels_nc1, 16)
    df_simdata['TBlabel'][(64 * 16 * 2):(64 * 16 * 3)] = np.repeat(labels_nc2, 16)
    df_simdata['TBlabel'][(64 * 16 * 3):(64 * 16 * 4)] = np.repeat(labels_nc3, 16)
    df_simdata['TBlabel'][(64 * 16 * 4):(64 * 16 * 5)] = np.repeat(labels_nc4, 16)

    return df_simdata


df_simdata_hyp1 = makedf(hyp1_f, hyp1_f_c1, hyp1_f_c2, hyp1_f_c3, hyp1_f_c4, labels_TB, labels_TB_nc1, labels_TB_nc2,
                         labels_TB_nc3, labels_TB_nc4, idx_f)
df_simdata_hyp2 = makedf(hyp2_f, hyp2_f_c1, hyp2_f_c2, hyp2_f_c3, hyp2_f_c4, labels_TB, labels_TB_nc1, labels_TB_nc2,
                         labels_TB_nc3, labels_TB_nc4, idx_f)
df_simdata_hyp3 = makedf(hyp3_f, hyp3_f_c1, hyp3_f_c2, hyp3_f_c3, hyp3_f_c4, labels_TB, labels_TB_nc1, labels_TB_nc2,
                         labels_TB_nc3, labels_TB_nc4, idx_f)

df_simdata_hyp1nf = makedf(hyp1_nf, hyp1_nf_c1, hyp1_nf_c2, hyp1_nf_c3, hyp1_nf_c4, labels_TB_nf, labels_TB_nf_nc1,
                           labels_TB_nf_nc2, labels_TB_nf_nc3, labels_TB_nf_nc4, idx_nf)
df_simdata_hyp2nf = makedf(hyp2_nf, hyp2_nf_c1, hyp2_nf_c2, hyp2_nf_c3, hyp2_nf_c4, labels_TB_nf, labels_TB_nf_nc1,
                           labels_TB_nf_nc2, labels_TB_nf_nc3, labels_TB_nf_nc4, idx_nf)
df_simdata_hyp3nf = makedf(hyp3_nf, hyp3_nf_c1, hyp3_nf_c2, hyp3_nf_c3, hyp3_nf_c4, labels_TB_nf, labels_TB_nf_nc1,
                           labels_TB_nf_nc2, labels_TB_nf_nc3, labels_TB_nf_nc4, idx_nf)

# FIGURE 6

x = range(12)
y = range(12)
plt.rc('text', usetex=True)
fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(15, 10), sharey=True)

sns.lineplot(data=df_simdata_hyp1[df_simdata_hyp1['TBlabel'] == 'Top'], x="SelfScore", y="OtherScore",
             ax=ax[0, 0], linewidth=2, color='deepskyblue')
ax[0, 0].scatter(np.linspace(0, 12, 13), df_simdata_hyp1['OtherScore'][df_simdata_hyp1['TBlabel'] == 'Top'].groupby(
    df_simdata_hyp1['SelfScore']).mean().values,
                 color='deepskyblue', marker="o")
sns.lineplot(data=df_simdata_hyp2[df_simdata_hyp2['TBlabel'] == 'Top'], x="SelfScore", y="OtherScore", ax=ax[0, 1],
             linewidth=2, color='deepskyblue')
ax[0, 1].scatter(np.linspace(0, 12, 13), df_simdata_hyp2['OtherScore'][df_simdata_hyp2['TBlabel'] == 'Top'].groupby(
    df_simdata_hyp2['SelfScore']).mean().values,
                 color='deepskyblue', marker="o")
sns.lineplot(data=df_simdata_hyp3[df_simdata_hyp3['TBlabel'] == 'Top'], x="SelfScore", y="OtherScore", ax=ax[0, 2],
             linewidth=2, color='deepskyblue')
ax[0, 2].scatter(np.linspace(0, 12, 13), df_simdata_hyp3['OtherScore'][df_simdata_hyp3['TBlabel'] == 'Top'].groupby(
    df_simdata_hyp3['SelfScore']).mean().values,
                 color='deepskyblue', marker="o")

sns.lineplot(data=df_simdata_hyp1[df_simdata_hyp1['TBlabel'] == 'Bottom'], x="SelfScore", y="OtherScore", ax=ax[0, 0],
             linewidth=2, color='darkorange')
ax[0, 0].scatter(np.linspace(0, 12, 13), df_simdata_hyp1['OtherScore'][df_simdata_hyp1['TBlabel'] == 'Bottom'].groupby(
    df_simdata_hyp1['SelfScore']).mean().values,
                 color='darkorange', marker="s")
sns.lineplot(data=df_simdata_hyp2[df_simdata_hyp2['TBlabel'] == 'Bottom'], x="SelfScore", y="OtherScore", ax=ax[0, 1],
             linewidth=2, color='darkorange')
ax[0, 1].scatter(np.linspace(0, 12, 13), df_simdata_hyp2['OtherScore'][df_simdata_hyp2['TBlabel'] == 'Bottom'].groupby(
    df_simdata_hyp2['SelfScore']).mean().values,
                 color='darkorange', marker="s")
sns.lineplot(data=df_simdata_hyp3[df_simdata_hyp3['TBlabel'] == 'Bottom'], x="SelfScore", y="OtherScore", ax=ax[0, 2],
             linewidth=2, color='darkorange')
ax[0, 2].scatter(np.linspace(0, 12, 13), df_simdata_hyp3['OtherScore'][df_simdata_hyp3['TBlabel'] == 'Bottom'].groupby(
    df_simdata_hyp3['SelfScore']).mean().values,
                 color='darkorange', marker="s")

sns.lineplot(data=df_simdata_hyp1nf, x="SelfScore", y="OtherScore",
             ax=ax[1, 0], linewidth=2, color='limegreen')
ax[1, 0].scatter(np.linspace(0, 12, 13),
                 df_simdata_hyp1nf['OtherScore'].groupby(df_simdata_hyp1nf['SelfScore']).mean().values,
                 color='limegreen', marker="D")
sns.lineplot(data=df_simdata_hyp2nf, x="SelfScore", y="OtherScore",
             ax=ax[1, 1], linewidth=2, color='limegreen')
ax[1, 1].scatter(np.linspace(0, 12, 13),
                 df_simdata_hyp2nf['OtherScore'].groupby(df_simdata_hyp2nf['SelfScore']).mean().values,
                 color='limegreen', marker="D")
sns.lineplot(data=df_simdata_hyp3nf, x="SelfScore", y="OtherScore",
             ax=ax[1, 2], linewidth=2, color='limegreen')
ax[1, 2].scatter(np.linspace(0, 12, 13),
                 df_simdata_hyp3nf['OtherScore'].groupby(df_simdata_hyp3nf['SelfScore']).mean().values,
                 color='limegreen', marker="D")

ax[0, 0].set_ylabel("Estimated Other Score", fontsize=14)
ax[1, 0].set_ylabel("Estimated Other Score", fontsize=14)

ax[0, 0].set_title(r"$M_1$: Fully Differentiated", fontsize=15, fontweight='bold')
ax[0, 1].set_title(r"$M_2$: Differentiated by Ability", fontsize=15, fontweight='bold')
ax[0, 2].set_title(r"$M_3$: Undifferentiated", fontsize=15, fontweight='bold')

for i in range(2):
    for j in range(3):
        ax[i, j].spines['right'].set_visible(False)
        ax[i, j].spines['top'].set_visible(False)
        ax[i, j].set_ylim(-1, 13)
        ax[i, j].set_xlim(-1, 13)
        ax[i, j].set_xlabel("Estimated Self Score", fontsize=14)

for row in ax:
    for col in row:
        col.plot(x, y, c='k', linestyle='dashed')

blue = mpatches.Patch(color='deepskyblue', label='Top')
orange = mpatches.Patch(color='darkorange', label='Bottom')
green = mpatches.Patch(color='limegreen', label='N/A')
fig.legend(handles=[blue, orange, green], bbox_to_anchor=(.9, .68), loc="center", borderaxespad=0., frameon=False,
           fontsize=14)

ax[0, 0].text(-.18, .7, 'Feedback', transform=ax[0, 0].transAxes, fontsize=16, fontweight='bold', va='top', ha='right',
              rotation=90)
ax[1, 0].text(-.18, .75, 'No Feedback', transform=ax[1, 0].transAxes, fontsize=16, fontweight='bold', va='top',
              ha='right', rotation=90)

plt.plot()
fig.savefig("/Users/aakritikumar/Desktop/Lab/ToM-pycharm/fig6.pdf", bbox_inches='tight')

# FIGURE 7


x = range(12)
y = range(12)

fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(15, 10), sharey=True)
sns.lineplot(data=df_simdata_hyp1[df_simdata_hyp1['TBlabel'] == 'Top'], x="Blocks", y="OtherScore", ax=ax[0, 0],
             linewidth=2, color='deepskyblue')
ax[0, 0].scatter(np.linspace(1, 16, 16), df_simdata_hyp1['OtherScore'][df_simdata_hyp1['TBlabel'] == 'Top'].groupby(
    df_simdata_hyp1['Blocks']).mean().values,
                 color='deepskyblue', marker="o")
sns.lineplot(data=df_simdata_hyp2[df_simdata_hyp2['TBlabel'] == 'Top'], x="Blocks", y="OtherScore", ax=ax[0, 1],
             linewidth=2, color='deepskyblue')
ax[0, 1].scatter(np.linspace(1, 16, 16), df_simdata_hyp2['OtherScore'][df_simdata_hyp2['TBlabel'] == 'Top'].groupby(
    df_simdata_hyp2['Blocks']).mean().values,
                 color='deepskyblue', marker="o")
sns.lineplot(data=df_simdata_hyp3[df_simdata_hyp3['TBlabel'] == 'Top'], x="Blocks", y="OtherScore", ax=ax[0, 2],
             linewidth=2, color='deepskyblue')
ax[0, 2].scatter(np.linspace(1, 16, 16), df_simdata_hyp3['OtherScore'][df_simdata_hyp3['TBlabel'] == 'Top'].groupby(
    df_simdata_hyp3['Blocks']).mean().values,
                 color='deepskyblue', marker="o")

sns.lineplot(data=df_simdata_hyp1nf[df_simdata_hyp1nf['TBlabel'] == 'Top'], x="Blocks", y="OtherScore", ax=ax[1, 0],
             linewidth=2, color='deepskyblue')
ax[1, 0].scatter(np.linspace(1, 16, 16), df_simdata_hyp1nf['OtherScore'][df_simdata_hyp1nf['TBlabel'] == 'Top'].groupby(
    df_simdata_hyp1nf['Blocks']).mean().values,
                 color='deepskyblue', marker="o")
sns.lineplot(data=df_simdata_hyp2nf[df_simdata_hyp2nf['TBlabel'] == 'Top'], x="Blocks", y="OtherScore", ax=ax[1, 1],
             linewidth=2, color='deepskyblue')
ax[1, 1].scatter(np.linspace(1, 16, 16), df_simdata_hyp2nf['OtherScore'][df_simdata_hyp2nf['TBlabel'] == 'Top'].groupby(
    df_simdata_hyp2nf['Blocks']).mean().values,
                 color='deepskyblue', marker="o")
sns.lineplot(data=df_simdata_hyp3nf[df_simdata_hyp3nf['TBlabel'] == 'Top'], x="Blocks", y="OtherScore", ax=ax[1, 2],
             linewidth=2, color='deepskyblue')
ax[1, 2].scatter(np.linspace(1, 16, 16), df_simdata_hyp3nf['OtherScore'][df_simdata_hyp3nf['TBlabel'] == 'Top'].groupby(
    df_simdata_hyp3nf['Blocks']).mean().values,
                 color='deepskyblue', marker="o")

sns.lineplot(data=df_simdata_hyp1[df_simdata_hyp1['TBlabel'] == 'Bottom'], x="Blocks", y="OtherScore", ax=ax[0, 0],
             linewidth=2, color='darkorange')
ax[0, 0].scatter(np.linspace(1, 16, 16), df_simdata_hyp1['OtherScore'][df_simdata_hyp1['TBlabel'] == 'Bottom'].groupby(
    df_simdata_hyp1['Blocks']).mean().values,
                 color='darkorange', marker="s")
sns.lineplot(data=df_simdata_hyp2[df_simdata_hyp2['TBlabel'] == 'Bottom'], x="Blocks", y="OtherScore", ax=ax[0, 1],
             linewidth=2, color='darkorange')
ax[0, 1].scatter(np.linspace(1, 16, 16), df_simdata_hyp2['OtherScore'][df_simdata_hyp2['TBlabel'] == 'Bottom'].groupby(
    df_simdata_hyp2['Blocks']).mean().values,
                 color='darkorange', marker="s")
sns.lineplot(data=df_simdata_hyp3[df_simdata_hyp3['TBlabel'] == 'Bottom'], x="Blocks", y="OtherScore", ax=ax[0, 2],
             linewidth=2, color='darkorange')
ax[0, 2].scatter(np.linspace(1, 16, 16), df_simdata_hyp3['OtherScore'][df_simdata_hyp3['TBlabel'] == 'Bottom'].groupby(
    df_simdata_hyp3['Blocks']).mean().values,
                 color='darkorange', marker="s")

sns.lineplot(data=df_simdata_hyp1nf[df_simdata_hyp1nf['TBlabel'] == 'Bottom'], x="Blocks", y="OtherScore", ax=ax[1, 0],
             linewidth=2, color='darkorange')
ax[1, 0].scatter(np.linspace(1, 16, 16),
                 df_simdata_hyp1nf['OtherScore'][df_simdata_hyp1nf['TBlabel'] == 'Bottom'].groupby(
                     df_simdata_hyp1nf['Blocks']).mean().values,
                 color='darkorange', marker="s")
sns.lineplot(data=df_simdata_hyp2nf[df_simdata_hyp2nf['TBlabel'] == 'Bottom'], x="Blocks", y="OtherScore", ax=ax[1, 1],
             linewidth=2, color='darkorange')
ax[1, 1].scatter(np.linspace(1, 16, 16),
                 df_simdata_hyp2nf['OtherScore'][df_simdata_hyp2nf['TBlabel'] == 'Bottom'].groupby(
                     df_simdata_hyp2nf['Blocks']).mean().values,
                 color='darkorange', marker="s")
sns.lineplot(data=df_simdata_hyp3nf[df_simdata_hyp3nf['TBlabel'] == 'Bottom'], x="Blocks", y="OtherScore", ax=ax[1, 2],
             linewidth=2, color='darkorange')
ax[1, 2].scatter(np.linspace(1, 16, 16),
                 df_simdata_hyp3nf['OtherScore'][df_simdata_hyp3nf['TBlabel'] == 'Bottom'].groupby(
                     df_simdata_hyp3nf['Blocks']).mean().values,
                 color='darkorange', marker="s")

ax[0, 0].set_ylabel("Mean Other Score", fontsize=14)
ax[1, 0].set_ylabel("Mean Other Score", fontsize=14)

ax[0, 0].set_title(r"$M_1$: Fully Differentiated", fontsize=15, fontweight='bold')
ax[0, 1].set_title(r"$M_2$: Differentiated by Ability", fontsize=15, fontweight='bold')
ax[0, 2].set_title(r"$M_3$: Undifferentiated", fontsize=15, fontweight='bold')

for i in range(2):
    for j in range(3):
        ax[i, j].spines['right'].set_visible(False)
        ax[i, j].spines['top'].set_visible(False)
        ax[i, j].set_ylim(3, 10)
        ax[i, j].set_xlim(1, 16)
        ax[i, j].set_xlabel("Problem Set", fontsize=14)
        ax[0, j].axhline(actcorrectothermean_top, linestyle='dashed', color='deepskyblue')
        ax[0, j].axhline(actcorrectothermean_bottom, linestyle='dashed', color='darkorange')
        ax[1, j].axhline(actcorrectothermean_top_nfb, linestyle='dashed', color='deepskyblue')
        ax[1, j].axhline(actcorrectothermean_bottom_nfb, linestyle='dashed', color='darkorange')

blue = mpatches.Patch(color='deepskyblue', label='Top')
orange = mpatches.Patch(color='darkorange', label='Bottom')
# green= mpatches.Patch(color='limegreen', label='N/A')
fig.legend(handles=[blue, orange], bbox_to_anchor=(.9, .45), loc="center", borderaxespad=0., frameon=False, fontsize=14)

ax[0, 0].text(-.18, .6, 'Feedback', transform=ax[0, 0].transAxes, fontsize=16, fontweight='bold', va='top', ha='right',
              rotation=90)
ax[1, 0].text(-.18, .65, 'No Feedback', transform=ax[1, 0].transAxes, fontsize=16, fontweight='bold', va='top',
              ha='right', rotation=90)

plt.plot()
fig.savefig("/Users/aakritikumar/Desktop/Lab/ToM-pycharm/fig7.pdf", bbox_inches='tight')
