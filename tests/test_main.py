def test_main_prints_greeting(main_module, capsys):
    main_module.main()
    assert capsys.readouterr().out == "Hello from full-stack-genai-bootcamp-1-0!\n"


def test_main_returns_none(main_module):
    assert main_module.main() is None
