import pandas as pd
import numpy as np
import os
from datetime import datetime
from scipy import stats
from scipy.stats import t, pearsonr, f_oneway, ttest_ind

# Настройка отображения
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# =============================================================================
# 1. Загрузка данных из data.csv
# =============================================================================
FILE_PATH = 'data.csv'
OUTPUT_EXCEL = 'sales_analysis_results.xlsx'

def get_output_filename():
    if not os.path.exists(OUTPUT_EXCEL):
        return OUTPUT_EXCEL
    try:
        with open(OUTPUT_EXCEL, 'ab'):
            pass
        return OUTPUT_EXCEL
    except PermissionError:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        new_name = f'sales_analysis_results_{timestamp}.xlsx'
        print(f"Файл {OUTPUT_EXCEL} занят. Сохраняем как {new_name}")
        return new_name

OUTPUT_EXCEL = get_output_filename()

df = pd.read_csv(FILE_PATH, delimiter=',', encoding='utf-8-sig')
df.columns = df.columns.str.lower()
df['dr_dat'] = pd.to_datetime(df['dr_dat'], format='%Y-%m-%d')
df['dr_kol'] = pd.to_numeric(df['dr_kol'], errors='coerce')
df['dr_czak'] = pd.to_numeric(df['dr_czak'], errors='coerce')
df['dr_croz'] = pd.to_numeric(df['dr_croz'], errors='coerce')
df.dropna(subset=['dr_cdrugs', 'dr_kol', 'dr_dat'], inplace=True)

print(f"Загружено записей: {len(df)}")
print(f"Период: с {df['dr_dat'].min()} по {df['dr_dat'].max()}")

# =============================================================================
# 2. Атрибуты товаров
# =============================================================================
product_name = df.groupby('dr_cdrugs')['dr_ndrugs'].first().to_dict()
supplier = df.groupby('dr_cdrugs')['dr_suppl'].first().to_dict()
manufacturer = df.groupby('dr_cdrugs')['dr_prod'].first().to_dict()
avg_price = df.groupby('dr_cdrugs')['dr_czak'].mean().to_dict()
avg_retail = df.groupby('dr_cdrugs')['dr_croz'].mean().to_dict()

# =============================================================================
# 3. Агрегация продаж и прибыли по дням и товарам
# =============================================================================
daily_agg = df.groupby(['dr_cdrugs', 'dr_dat']).agg(
    total_kol=('dr_kol', 'sum'),
    total_profit=('dr_kol', lambda x: ((df.loc[x.index, 'dr_croz'] - df.loc[x.index, 'dr_czak']) * x).sum())
).reset_index()
daily_agg.rename(columns={'total_kol': 'dr_kol', 'total_profit': 'profit'}, inplace=True)

pivot_kol = daily_agg.pivot(index='dr_cdrugs', columns='dr_dat', values='dr_kol').fillna(0)
pivot_profit = daily_agg.pivot(index='dr_cdrugs', columns='dr_dat', values='profit').fillna(0)
products = pivot_kol.index.unique()
print(f"Уникальных товаров с продажами: {len(products)}")

# =============================================================================
# 4. Расчёт статистик по каждому товару
# =============================================================================
stats_list = []
for prod in products:
    sales_series = pivot_kol.loc[prod].values
    profit_series = pivot_profit.loc[prod].values
    
    n_days = len(sales_series)
    mean_day = np.mean(sales_series)
    std_day = np.std(sales_series, ddof=1) if n_days > 1 else 0
    max_day = np.max(sales_series)
    total_sales = np.sum(sales_series)
    n_sales_days = np.sum(sales_series > 0)
    cv = std_day / mean_day if mean_day > 0 else np.nan
    
    total_profit = np.sum(profit_series)
    avg_profit_per_day = np.mean(profit_series)
    std_profit = np.std(profit_series, ddof=1) if n_days > 1 else 0
    
    price = avg_price.get(prod, np.nan)
    retail = avg_retail.get(prod, np.nan)
    margin_per_unit = retail - price if not np.isnan(retail) and not np.isnan(price) else np.nan
    
    stats_list.append({
        'product_code': prod,
        'product_name': product_name.get(prod, ''),
        'supplier': supplier.get(prod, ''),
        'manufacturer': manufacturer.get(prod, ''),
        'avg_price': price,
        'avg_retail': retail,
        'margin_per_unit': margin_per_unit,
        'n_days': n_days,
        'n_sales_days': n_sales_days,
        'mean_day': mean_day,
        'std_day': std_day,
        'max_day': max_day,
        'total_sales': total_sales,
        'total_profit': total_profit,
        'avg_profit_per_day': avg_profit_per_day,
        'std_profit': std_profit,
        'cv': cv
    })

stats_df = pd.DataFrame(stats_list)
stats_df = stats_df[stats_df['total_sales'] > 0].reset_index(drop=True)

# =============================================================================
# 5. Расчёт дневного и недельного запаса (прогнозный интервал 95%)
# =============================================================================
def calc_day_stock(row):
    n = row['n_sales_days']
    mean = row['mean_day']
    std = row['std_day']
    maxd = row['max_day']
    if n >= 3 and std > 0:
        t_val = t.ppf(0.975, df=n-1)
        pi_upper = mean + t_val * std * np.sqrt(1 + 1/n)
        return np.ceil(pi_upper)
    else:
        return np.ceil(1.5 * maxd)

def calc_week_stock(row):
    n = row['n_sales_days']
    mean = row['mean_day']
    std = row['std_day']
    maxd = row['max_day']
    if n >= 3 and std > 0:
        t_val = t.ppf(0.975, df=n-1)
        upper_week = 7 * mean + t_val * np.sqrt(7) * std * np.sqrt(1 + 1/n)
        return np.ceil(upper_week)
    else:
        return np.ceil(maxd * 7 * 1.5)

stats_df['day_stock'] = stats_df.apply(calc_day_stock, axis=1)
stats_df['week_stock'] = stats_df.apply(calc_week_stock, axis=1)

# =============================================================================
# 6. Группа стабильности по CV
# =============================================================================
def cv_category(cv):
    if pd.isna(cv):
        return 'C (недостаточно данных)'
    elif cv < 0.5:
        return 'A (стабильные)'
    elif cv < 1.0:
        return 'B (умеренные)'
    else:
        return 'C (волатильные)'

stats_df['cv_group'] = stats_df['cv'].apply(cv_category)

# =============================================================================
# 7. ABC-анализ по прибыли и по объёму
# =============================================================================
def abc_classify(df, column, label_prefix):
    sorted_df = df.sort_values(by=column, ascending=False).reset_index(drop=True)
    total = sorted_df[column].sum()
    cumsum = sorted_df[column].cumsum() / total
    classes = []
    for val in cumsum:
        if val <= 0.8:
            classes.append(f'{label_prefix} A (высокий)')
        elif val <= 0.95:
            classes.append(f'{label_prefix} B (средний)')
        else:
            classes.append(f'{label_prefix} C (низкий)')
    sorted_df['abc_class'] = classes
    df[f'ABC_{column}'] = sorted_df.set_index(sorted_df.index)['abc_class']
    return df

stats_df = abc_classify(stats_df, 'total_profit', 'Прибыль')
stats_df = abc_classify(stats_df, 'total_sales', 'Объём')
stats_df.rename(columns={
    'ABC_total_profit': 'ABC (по прибыли)',
    'ABC_total_sales': 'ABC (по объёму)'
}, inplace=True)

# =============================================================================
# 8. Общая статистика и сводки
# =============================================================================
total_transactions = len(df)
unique_products = len(stats_df)
total_profit_all = stats_df['total_profit'].sum()
total_sales_all = stats_df['total_sales'].sum()

general_stats = pd.DataFrame({
    'Показатель': [
        'Всего транзакций',
        'Уникальных товаров с продажами',
        'Общий объём продаж, шт.',
        'Общая прибыль за период, руб.',
        'Среднее в день (по всем товарам), шт.',
        'Медианное в день, шт.',
        'Станд. отклонение дневных продаж (среднее по товарам)',
        'Доля дней с нулевыми продажами (в среднем)'
    ],
    'Значение': [
        total_transactions,
        unique_products,
        f"{total_sales_all:,.0f}",
        f"{total_profit_all:,.0f}",
        f"{stats_df['mean_day'].mean():.2f}",
        f"{stats_df['mean_day'].median():.2f}",
        f"{stats_df['std_day'].mean():.2f}",
        f"{1 - stats_df['n_sales_days'].mean() / stats_df['n_days'].mean():.2%}"
    ]
})

cv_distribution = stats_df['cv_group'].value_counts(normalize=True).sort_index().reset_index()
cv_distribution.columns = ['Группа стабильности', 'Доля']

abc_profit_dist = stats_df['ABC (по прибыли)'].value_counts(normalize=True).sort_index().reset_index()
abc_profit_dist.columns = ['Класс ABC (прибыль)', 'Доля']

abc_volume_dist = stats_df['ABC (по объёму)'].value_counts(normalize=True).sort_index().reset_index()
abc_volume_dist.columns = ['Класс ABC (объём)', 'Доля']

# Топ-10 по прибыли и по объёму
top10_profit = stats_df.nlargest(10, 'total_profit')[
    ['product_code', 'product_name', 'total_profit', 'total_sales', 'margin_per_unit', 'week_stock']
].reset_index(drop=True)

top10_volume = stats_df.nlargest(10, 'total_sales')[
    ['product_code', 'product_name', 'total_sales', 'total_profit', 'mean_day', 'cv', 'week_stock']
].reset_index(drop=True)

# =============================================================================
# 8.1 Дополнительная статистика по прибыли (новый лист)
# =============================================================================
profit_stats = pd.DataFrame({
    'Показатель': [
        'Общая прибыль за период, руб.',
        'Средняя прибыль на товар, руб.',
        'Медианная прибыль на товар, руб.',
        'Стандартное отклонение прибыли на товар, руб.',
        'Минимальная прибыль на товар, руб.',
        'Максимальная прибыль на товар, руб.',
        'Количество товаров с прибылью > 0',
        'Количество товаров с прибылью = 0',
        'Доля товаров с нулевой прибылью',
        '25-й процентиль (Q1), руб.',
        '75-й процентиль (Q3), руб.',
        'Коэффициент вариации прибыли (CV_profit)'
    ],
    'Значение': [
        f"{total_profit_all:,.2f}",
        f"{stats_df['total_profit'].mean():.2f}",
        f"{stats_df['total_profit'].median():.2f}",
        f"{stats_df['total_profit'].std():.2f}",
        f"{stats_df['total_profit'].min():.2f}",
        f"{stats_df['total_profit'].max():.2f}",
        f"{(stats_df['total_profit'] > 0).sum()}",
        f"{(stats_df['total_profit'] == 0).sum()}",
        f"{(stats_df['total_profit'] == 0).mean():.2%}",
        f"{stats_df['total_profit'].quantile(0.25):.2f}",
        f"{stats_df['total_profit'].quantile(0.75):.2f}",
        f"{stats_df['total_profit'].std() / stats_df['total_profit'].mean() if stats_df['total_profit'].mean() > 0 else np.nan:.2f}"
    ]
})

# Распределение прибыли по интервалам
profit_bins = [0, 10, 50, 100, 500, 1000, 5000, 10000, float('inf')]
labels = ['0-10', '10-50', '50-100', '100-500', '500-1000', '1000-5000', '5000-10000', '>10000']
stats_df['profit_group'] = pd.cut(stats_df['total_profit'], bins=profit_bins, labels=labels, right=False)
profit_distribution = stats_df['profit_group'].value_counts().sort_index().reset_index()
profit_distribution.columns = ['Интервал прибыли, руб.', 'Количество товаров']
profit_distribution['Доля'] = profit_distribution['Количество товаров'] / len(stats_df)

# Топ-5 и Bottom-5 по прибыли (для отдельных листов)
top5_profit = stats_df.nlargest(5, 'total_profit')[
    ['product_code', 'product_name', 'total_profit', 'total_sales']
].reset_index(drop=True)
bottom5_profit = stats_df.nsmallest(5, 'total_profit')[
    ['product_code', 'product_name', 'total_profit', 'total_sales']
].reset_index(drop=True)

# Сводка по ABC-классам прибыли
abc_profit_summary = stats_df.groupby('ABC (по прибыли)').agg(
    Количество_товаров=('product_code', 'count'),
    Суммарная_прибыль=('total_profit', 'sum'),
    Средняя_прибыль=('total_profit', 'mean'),
    Доля_в_общей_прибыли=('total_profit', lambda x: x.sum() / total_profit_all * 100)
).reset_index()
abc_profit_summary['Доля_товаров'] = abc_profit_summary['Количество_товаров'] / len(stats_df) * 100

# =============================================================================
# 9. Проверка гипотез (без изменений)
# =============================================================================
df['is_weekend'] = df['dr_dat'].dt.weekday.isin([5, 6])

groupA = stats_df[stats_df['cv_group'] == 'A (стабильные)']['product_code'].tolist()
weekday_sales, weekend_sales = [], []
for prod in groupA:
    prod_data = df[df['dr_cdrugs'] == prod]
    weekday_sales.extend(prod_data[~prod_data['is_weekend']]['dr_kol'].tolist())
    weekend_sales.extend(prod_data[prod_data['is_weekend']]['dr_kol'].tolist())
if weekday_sales and weekend_sales:
    t_stat, p_val = ttest_ind(weekday_sales, weekend_sales, equal_var=False)
    h1 = f"t={t_stat:.3f}, p={p_val:.4f} -> {'не значимы' if p_val>0.05 else 'значимы'}"
else:
    h1 = "недостаточно данных"

price_clean = stats_df[stats_df['avg_price'].notna() & (stats_df['n_sales_days'] > 0)]
if len(price_clean) > 1:
    corr, p_val = pearsonr(price_clean['avg_price'], price_clean['n_sales_days'])
    h2 = f"r={corr:.3f}, p={p_val:.4f} -> {'значимая' if p_val<0.05 else 'не значимая'}"
else:
    h2 = "недостаточно данных"

high = stats_df[stats_df['cv'] >= 1.0]['max_day']
low = stats_df[stats_df['cv'] < 0.5]['max_day']
if len(high)>0 and len(low)>0:
    t_stat, p_val = ttest_ind(high, low, equal_var=False)
    h3 = f"t={t_stat:.3f}, p={p_val:.4f} -> {'значимы' if p_val<0.05 else 'не значимы'}"
else:
    h3 = "недостаточно данных"

daily_total = df.groupby('dr_dat')['dr_kol'].sum().reset_index()
daily_total['weekday'] = daily_total['dr_dat'].dt.weekday
groups = [daily_total[daily_total['weekday'] == d]['dr_kol'].values for d in range(7)]
groups = [g for g in groups if len(g) > 0]
if len(groups) >= 2:
    f_stat, p_val = f_oneway(*groups)
    h4 = f"F={f_stat:.3f}, p={p_val:.4f} -> {'не значимы' if p_val>0.05 else 'значимы'}"
else:
    h4 = "недостаточно данных"

if len(price_clean) > 1:
    corr, p_val = pearsonr(price_clean['avg_price'], price_clean['total_sales'])
    h5 = f"r={corr:.3f}, p={p_val:.4f} -> {'значимая' if p_val<0.05 else 'не значимая'}"
else:
    h5 = "недостаточно данных"

cv_clean = stats_df[stats_df['cv'].notna() & (stats_df['total_sales'] > 0)]
if len(cv_clean) > 1:
    corr, p_val = pearsonr(cv_clean['total_sales'], cv_clean['cv'])
    h6 = f"r={corr:.3f}, p={p_val:.4f} -> {'значимая' if p_val<0.05 else 'не значимая'}"
else:
    h6 = "недостаточно данных"

hypotheses_df = pd.DataFrame({
    'Гипотеза': [
        'H1: Продажи стабильных товаров не зависят от дня недели',
        'H2: Цена отрицательно коррелирует с частотой продаж',
        'H3: Волатильные товары имеют более высокий максимум',
        'H4: Продажи равномерны по дням недели',
        'H5: Цена отрицательно коррелирует с объёмом продаж',
        'H6: Больший объём продаж → меньшая вариативность'
    ],
    'Результат': [h1, h2, h3, h4, h5, h6]
})

# =============================================================================
# 10. Формирование итоговой сводной таблицы (русские заголовки)
# =============================================================================
rename_map = {
    'product_code': 'Код товара',
    'product_name': 'Наименование товара',
    'supplier': 'Поставщик',
    'manufacturer': 'Производитель',
    'avg_price': 'Средняя закупочная цена, руб.',
    'avg_retail': 'Средняя розничная цена, руб.',
    'margin_per_unit': 'Маржа на ед., руб.',
    'total_sales': 'Общий объём продаж, шт.',
    'total_profit': 'Общая прибыль, руб.',
    'mean_day': 'Среднее в день, шт.',
    'std_day': 'Стандартное отклонение, шт.',
    'max_day': 'Максимум в день, шт.',
    'cv': 'Коэффициент вариации (CV)',
    'cv_group': 'Группа стабильности',
    'ABC (по прибыли)': 'ABC (по прибыли)',
    'ABC (по объёму)': 'ABC (по объёму)',
    'day_stock': 'Дневной запас (95%), шт.',
    'week_stock': 'Недельный запас (95%), шт.'
}

output_cols = ['product_code', 'product_name', 'supplier', 'manufacturer',
               'avg_price', 'avg_retail', 'margin_per_unit',
               'total_sales', 'total_profit', 'mean_day', 'std_day', 'max_day',
               'cv', 'cv_group', 'ABC (по прибыли)', 'ABC (по объёму)',
               'day_stock', 'week_stock']

stats_df_output = stats_df[output_cols].rename(columns=rename_map)

top10_profit_out = top10_profit.rename(columns={
    'product_code': 'Код товара',
    'product_name': 'Наименование товара',
    'total_profit': 'Общая прибыль, руб.',
    'total_sales': 'Общий объём продаж, шт.',
    'margin_per_unit': 'Маржа на ед., руб.',
    'week_stock': 'Недельный запас (95%), шт.'
})

top10_volume_out = top10_volume.rename(columns={
    'product_code': 'Код товара',
    'product_name': 'Наименование товара',
    'total_sales': 'Общий объём продаж, шт.',
    'total_profit': 'Общая прибыль, руб.',
    'mean_day': 'Среднее в день, шт.',
    'cv': 'Коэффициент вариации (CV)',
    'week_stock': 'Недельный запас (95%), шт.'
})

bottom5_profit_out = bottom5_profit.rename(columns={
    'product_code': 'Код товара',
    'product_name': 'Наименование товара',
    'total_profit': 'Прибыль, руб.',
    'total_sales': 'Объём продаж, шт.'
})

# =============================================================================
# 11. Сохранение в Excel (обновлено – удалены лишние вкладки)
# =============================================================================
with pd.ExcelWriter(OUTPUT_EXCEL, engine='openpyxl') as writer:
    general_stats.to_excel(writer, sheet_name='Общая статистика', index=False)
    profit_stats.to_excel(writer, sheet_name='Статистика прибыли', index=False)
    cv_distribution.to_excel(writer, sheet_name='Группы CV', index=False)
    abc_profit_dist.to_excel(writer, sheet_name='ABC (прибыль)', index=False)
    abc_volume_dist.to_excel(writer, sheet_name='ABC (объём)', index=False)
    
    profit_distribution.to_excel(writer, sheet_name='Распределение прибыли', index=False)
    bottom5_profit_out.to_excel(writer, sheet_name='Bottom-5 по прибыли', index=False)
    abc_profit_summary.to_excel(writer, sheet_name='ABC сводка по прибыли', index=False)
    
    top10_profit_out.to_excel(writer, sheet_name='Топ-10 по прибыли', index=False)
    top10_volume_out.to_excel(writer, sheet_name='Топ-10 по объёму', index=False)
    stats_df_output.to_excel(writer, sheet_name='Все товары', index=False)
    hypotheses_df.to_excel(writer, sheet_name='Гипотезы', index=False)

print(f"\n✅ Результаты сохранены в файл: {OUTPUT_EXCEL}")

# =============================================================================
# 12. Краткий вывод в консоль
# =============================================================================
print("\n=== Краткая сводка ===")
print(f"Всего товаров с продажами: {unique_products}")
print(f"Общая прибыль за период: {total_profit_all:,.0f} руб.")
print(f"Средняя прибыль на товар: {stats_df['total_profit'].mean():.2f} руб.")
print(f"Медианная прибыль на товар: {stats_df['total_profit'].median():.2f} руб.")
print(f"Общий объём продаж: {total_sales_all:,.0f} шт.")
print(f"Средний рекомендуемый недельный запас: {stats_df['week_stock'].mean():.1f} шт.")
print(f"Медианный рекомендуемый недельный запас: {stats_df['week_stock'].median():.0f} шт.")
print("\nГруппы стабильности:")
print(cv_distribution.to_string(index=False))
print("\nABC (по прибыли):")
print(abc_profit_dist.to_string(index=False))

print("\n✅ Анализ завершён!")