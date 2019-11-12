from matplotlib import pyplot
time_list = [2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017]
time_list_append = [2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022]
predict = [10453, 10655.758342548274, 10726.259909546003, 10651.26868581213, 10399.137368507683, 9965.11495705042, 9558.341245321557, 9299.306648404337, 9244.162675034255, 9467.18974905461, 10074.03154500015, 11225.023346560076, 13175.140982799232]
real = [10453, 10289, 10722, 11148, 11081, 9865, 9093, 9394]
pyplot.plot(time_list_append, predict)
pyplot.plot(time_list, real)
pyplot.xticks(time_list_append[::3])
pyplot.xlabel('Year')
pyplot.ylabel('{0}DrugReports'.format('All'))
pyplot.title('KY')
pyplot.legend(['predict', 'real'])
pyplot.show()