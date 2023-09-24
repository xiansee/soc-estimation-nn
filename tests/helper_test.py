from soc_estimation_nn.helper import XYData, plot


def test_XYData():
    """ Test XYData class intialization. """
    
    plot_settings = {
        'x_data': [1,2,3,4,5],
        'y_data': [2,2,2,2,2],
        'label': 'horizontal line',
        'color': 'black',
        'linestyle': '--',
    }
    settings_value = plot_settings.get

    data = XYData(**plot_settings)

    assert data.x_data == settings_value('x_data')
    assert data.y_data == settings_value('y_data')
    assert data.label == settings_value('label')
    assert data.color == settings_value('color')
    assert data.linestyle == settings_value('linestyle')
    assert data.plot_num == 1
    assert data.marker == ''


def test_plot():
    """ Test plot function. """

    fig, axes = plot(
        xy_data=[
            {
                'x_data': [1,2,3,4,5], 
                'y_data': [5,4,3,2,1], 
                'label': 'Line 1'
            },
            {
                'x_data': [1,2,3,4,5], 
                'y_data': [3,3,3,3,3], 
                'label': 'Line 2',
                'plot_num': 2
            }
        ], 
        x_label='X', 
        y_label={1: 'Y1', 2: 'Y2'}, 
        title='Foo', 
        show_plt=False
    )

    assert len(axes) == 2
    assert axes[0].get_xlabel() == 'X'
    assert axes[0].get_ylabel() == 'Y1'
    assert axes[1].get_ylabel() == 'Y2'
    assert fig._suptitle._text == 'Foo'
    assert all(axes[0].get_children()[0].get_xdata() == [1,2,3,4,5])
    assert all(axes[0].get_children()[0].get_ydata() == [5,4,3,2,1])
    assert all(axes[1].get_children()[0].get_ydata() == [3,3,3,3,3])

