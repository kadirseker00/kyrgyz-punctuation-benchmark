from kg_punctuation.translit import romanize


def test_kyrgyz_specific_letters_are_preserved():
    assert romanize("бирок эки жыл чыдап берсеңер, өнүгүү жолуна түшөбүз.") == (
        "birok eki jyl chydap berseñer, önügüü joluna tüshöbüz.".replace("ñ", "ng")
    )


def test_ye_rule_and_capitals():
    assert romanize("Ерлан") == "Yerlan"
    assert romanize("мекендештер") == "mekendeshter"
    assert romanize("журналисттер Бишкек") == "jurnalistter Bishkek"
    assert romanize("АКШ") == "AKSh"
