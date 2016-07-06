# When you restore the container model

model = (tfplus.nn.model
 .create_from_main('blah')                  # create sub-models in ctor.
 .set_folder('new_folder')                  # recursive set_folder()
 .restore_options_from('old_folder')        # recursive restore_options_from()
 .build_all()                               # recursive build()
 .init(sess)                                # already initialize inner models.
                                            # restore_weights_from options.
 .restore_weights_from(sess, 'old_folder')) # recursive restore weights

# When you initialize a container model fresh
nn.model
.create_from_main('blah')                   # create sub-models in ctor.
.set_folder('new_folder')                   # recursive set_folder()
.build_all()                                # recursive build()
.init(sess)                                 # restore_weights_from options
.restore_weights_from('old_folder')

# Initialize
nn.model
.create_from_main('blah')
.set_name('container')
.set_folder('new_folder')
.init(sess)
.restore_options_from('old_folder')
