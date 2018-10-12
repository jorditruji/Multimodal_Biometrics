


class Conv1DResBlock(nn.Module):

    def __init__(self, ninputs, fmaps, kwidth=3,
                 dilations=[1, 2, 4, 8], stride=4, bias=True,
                 transpose=False, act='prelu'):
        super(Conv1DResBlock, self).__init__()
        self.ninputs = ninputs
        self.fmaps = fmaps
        self.kwidth = kwidth
        self.dilations = dilations
        self.stride = stride
        self.bias = bias
        self.transpose = transpose
        assert dilations[0] == 1, dilations[0]
        assert len(dilations) > 1, len(dilations)
        self.convs = nn.ModuleList()
        self.acts = nn.ModuleList()
        prev_in = ninputs
        for n, d in enumerate(dilations):
            if n == 0:
                curr_stride = stride
            else:
                curr_stride = 1
            if n == 0 or (n + 1) >= len(dilations):
                # in the interfaces in/out it is different
                curr_fmaps = fmaps
            else:
                curr_fmaps = fmaps // 4
                assert curr_fmaps > 0, curr_fmaps
            if n == 0 and transpose:
                p_ = (self.kwidth - 4)//2
                op_ = 0
                if p_ < 0:
                    op_ = p_ * -1
                    p_ = 0
                self.convs.append(nn.ConvTranspose1d(prev_in, curr_fmaps, kwidth,
                                                     stride=curr_stride,
                                                     dilation=d,
                                                     padding=p_,
                                                     output_padding=op_,
                                                     bias=bias))
            else:
                self.convs.append(nn.Conv1d(prev_in, curr_fmaps, kwidth,
                                            stride=curr_stride,
                                            dilation=d,
                                            padding=0,
                                            bias=bias))
            self.acts.append(nn.PReLU(curr_fmaps))
            prev_in = curr_fmaps

    def forward(self, x):
        h = x
        res_act = None
        for li, layer in enumerate(self.convs):
            if self.stride > 1 and li == 0:
                # add proper padding
                pad_tuple = ((self.kwidth//2)-1, self.kwidth//2)
            else:
                # symmetric padding
                p_ = ((self.kwidth - 1) * self.dilations[li]) // 2
                pad_tuple = (p_, p_)
            #print('Applying pad tupple: ', pad_tuple)
            if not (self.transpose and li == 0):
                h = F.pad(h, pad_tuple)
            #print('Layer {}'.format(li))
            #print('h padded: ', h.size())
            h = layer(h)
            h = self.acts[li](h)
            if li == 0:
                # keep the residual activation
                res_act = h
            #print('h min: ', h.min())
            #print('h max: ', h.max())
            #print('h conved size: ', h.size())
        # add the residual activation in the output of the module
        return h + res_act


class GBlock(nn.Module):

    def __init__(self, ninputs, fmaps, kwidth,
                 activation, padding=None,
                 lnorm=False, dropout=0.,
                 pooling=2, enc=True, bias=False,
                 aal_h=None, linterp=False, snorm=False,
                 convblock=False):
        # linterp: do linear interpolation instead of simple conv transpose
        # snorm: spectral norm
        super(GBlock, self).__init__()
        self.pooling = pooling
        self.linterp = linterp
        self.enc = enc
        self.kwidth = kwidth
        self.convblock= convblock
        if padding is None:
            padding = 0
        if enc:
            if aal_h is not None:
                self.aal_conv = nn.Conv1d(ninputs, ninputs,
                                          aal_h.shape[0],
                                          stride=1,
                                          padding=aal_h.shape[0] // 2 - 1,
                                          bias=False)
                if snorm:
                    self.aal_conv = SpectralNorm(self.aal_conv)
                # apply AAL weights, reshaping impulse response to match
                # in channels and out channels
                aal_t = torch.FloatTensor(aal_h).view(1, 1, -1)
                aal_t = aal_t.repeat(ninputs, ninputs, 1)
                self.aal_conv.weight.data = aal_t
            if convblock:
                self.conv = Conv1DResBlock(ninputs, fmaps, kwidth,
                                           stride=pooling, bias=bias)
            else:
                self.conv = nn.Conv1d(ninputs, fmaps, kwidth,
                                      stride=pooling,
                                      padding=padding,
                                      bias=bias)
            if snorm:
                self.conv = SpectralNorm(self.conv)
            if activation == 'glu':
                # TODO: REVIEW
                raise NotImplementedError
                self.glu_conv = nn.Conv1d(ninputs, fmaps, kwidth,
                                          stride=pooling,
                                          padding=padding,
                                          bias=bias)
                if snorm:
                    self.glu_conv = spectral_norm(self.glu_conv)
        else:
            if linterp:
                # pre-conv prior to upsampling
                self.pre_conv = nn.Conv1d(ninputs, ninputs // 8,
                                          kwidth, stride=1, padding=kwidth//2,
                                          bias=bias)
                self.conv = nn.Conv1d(ninputs // 8, fmaps, kwidth,
                                      stride=1, padding=kwidth//2,
                                      bias=bias)
                if snorm:
                    self.conv = SpectralNorm(self.conv)
                if activation == 'glu':
                    self.glu_conv = nn.Conv1d(ninputs, fmaps, kwidth,
                                              stride=1, padding=kwidth//2,
                                              bias=bias)
                    if snorm:
                        self.glu_conv = SpectralNorm(self.glu_conv)
            else:
                if convblock:
                    self.conv = Conv1DResBlock(ninputs, fmaps, kwidth,
                                               stride=pooling, bias=bias,
                                               transpose=True)
                else:
                    # decoder like with transposed conv
                    # compute padding required based on pooling
                    pad = (2 * pooling - pooling - kwidth)//-2
                    self.conv = nn.ConvTranspose1d(ninputs, fmaps, kwidth,
                                                   stride=pooling,
                                                   padding=pad,
                                                   output_padding=0,
                                                   bias=bias)
                if snorm:
                    self.conv = SpectralNorm(self.conv)
                if activation == 'glu':
                    # TODO: REVIEW
                    raise NotImplementedError
                    self.glu_conv = nn.ConvTranspose1d(ninputs, fmaps, kwidth,
                                                       stride=pooling,
                                                       padding=padding,
                                                       output_padding=pooling-1,
                                                       bias=bias)
                    if snorm:
                        self.glu_conv = spectral_norm(self.glu_conv)
        if activation is not None:
            self.act = activation
        if lnorm:
            self.ln = LayerNorm()
        if dropout > 0:
            self.dout = nn.Dropout(dropout)

    def forward(self, x):

        if len(x.size()) == 4:
            print(type(x.data))
            # inverse case from 1D -> 2D, go 2D -> 1D
            # re-format input from [B, K, C, L] to [B, K * C, L]
            # where C: frequency, L: time
            x = x.squeeze(1)
        if hasattr(self, 'aal_conv'):
            x = self.aal_conv(x)
        if self.linterp:
            x = self.pre_conv(x)
            x = F.upsample(x, scale_factor=self.pooling,
                           mode='linear', align_corners=True)
        if self.enc:
            # apply proper padding
            x = F.pad(x, ((self.kwidth//2)-1, self.kwidth//2))
        #print(x.data.shape)
        h = self.conv(x)
        #print(type(h.data))
        if not self.enc and not self.linterp and not self.convblock:
            # trim last value of h perque el kernel es imparell
            # TODO: generalitzar a kernel parell/imparell
            #print('h size: ', h.size())
            h = h[:, :, :-1]
        linear_h = h
        #print(type(linear_h.data))
        #print(type(h.data))
        if hasattr(self, 'act'):
            if self.act == 'glu':
                hg = self.glu_conv(x)
                h = h * F.sigmoid(hg)
            else:
                h = self.act(h)
        if hasattr(self, 'ln'):
            h = self.ln(h)
        if hasattr(self, 'dout'):
            h = self.dout(h)
        #print(type(h.data))
        return h, linear_h
