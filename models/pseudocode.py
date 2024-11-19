def compute_detr_inputs(self, features, samples, pos):
        '''
        Objective: takes the features obtained from the backbone network and transforms them channel-wise by applying a linear layer

        output: only the features are transformed channel-wise by a linear layer([[a1,a2],[b1,b2]] -> [[c*a1,c*a2],[c*b1,c*b2]]), mask and pos remain as it is
        Output shape:
        srcs - (bs, 256, h, w)
        masks - (bs, h, w)
        pos - (bs, 256, h, w)
        query_embeds - (bs,100,512)
        '''
        srcs = []
        masks = []
        for l, feat in enumerate(features): #features -> List[NestedTensor] shape: (bs, 256, 60, 94)
            src, mask = feat.decompose()  #src(original feature) -> tensor of shape (bs, 256, 60, 94) and mask (torch.zeros) -> tensor of shape (bs, 60, 94)
            srcs.append(self.input_proj[l](src)) #input_proj is a linear NN layer that converts (bs, c, h, w) to (bs, 256, h, w) and then does groupnorm 
            masks.append(mask)
            assert mask is not None

        query_embeds = self.query_embed.weight # creates an embedding from 100 dim to 512

        return srcs, masks, pos, query_embeds


class transformer(srcs, masks, pos, query_embeds):

    # MSDeformAttn class (/models/ops/modules/ms_deform_attn.py) [USED IN ENCODER AND DECODER]
    def def_attn(query, reference_points, input_flatten, input_spatial_shapes):
        '''
        (Def-DETR Figure 2 and Equation 2) 
        Implementation:
        query_feat - z_q
        reference_pt - p_q
        feature_map - x

        for the qth reference pt
        ------------------------
        A_{mqk} - attention weight of the kth sampling pt in the mth attention head
        \Delta p_{mqk} -  sampling offset of the kth sampling pt in the mth attention head
        p_q + \Delta p_{mqk} - sampling locations (vary k and m to generate all locs)

        FINAL EQUATION
        ---------------
        DefAttn(z_q, p_q, x) = \sum_{m = 1 to #Atten-Heads} Wm [\sum_{k=1 to #Sampled-Locs} A_{mqk} * W'm * x(p_q + \Delta p_{mqk})]
        
        '''
        batch_size,concat_wh,num_channels = input_flatten.shape
        n_heads = 8 # number of attention heads
        
        value = LinearTransformation(input_flatten) # Linear NN layer
        value = value.view(batch_size, concat_wh, n_heads, num_channels // n_heads) # divides the num_channels equally among n_heads (eg. 256 channels -> 32 channels for each of the 8 heads)
        sampling_offsets = LinearTransformation(query)
        attention_weights = Softmax(LinearTransformation(query))

        offset_normalizer = torch.stack([input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1)
        sampling_locations = reference_points + (sampling_offsets / offset_normalizer)

        output = Aggregate(value, sampling_locations, attention_weights) # sum( value[sampling_locations] * attention_weights )
        output = LinearTransformation(output)
        return output


    def get_reference_points(spatial_shapes, valid_ratios, device):
        '''
        Given a feature map of shape bs, H,W:
         - for each img in the batch, it generates 2D locations in the range [0.5 to (H-0.5)] x [0.5 to (W-0.5)] i.e. a total of H*W pts
         - normalizes the generated points so that values of the locs lie in the range [0 - 1.0]
        '''
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):

            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                                        torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]


    def forward(self, srcs, masks, pos_embeds, query_embed=None):
        '''
        src: (bs,c,h,w)
        masks: (bs,h,w) - (padding_mask=> padded pixels have 1, others 0)
        pos_embeds: (bs,c,h,w)
        '''

        # prepare input for encoder
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            bs, c, h, w = src.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            src = src.flatten(2).transpose(1, 2) # shape (2,5640,256) (note: h*w = 5640)
            mask = mask.flatten(1) # shape (2,5640)

            ## To investigate further
            pos_embed = pos_embed.flatten(2).transpose(1, 2) # shape (2,5640,256) (note: h*w = 5640)
        
            # level_embed?
            '''
            (Pg 6 in paper)
            In application of the multi-scale deformable attention module in encoder, the output are of multi-
            scale feature maps with the same resolutions as the input. Both the key and query elements are
            of pixels from the multi-scale feature maps. For each query pixel, the reference point is itself. To
            identify which feature level each query pixel lies in, we add a scale-level embedding, denoted as e l ,
            to the feature representation, in addition to the positional embedding.
            '''
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1) # same shape as pos_embed

            # appends
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)


        # flatten lists
        src_flatten = torch.cat(src_flatten, 1) # concat along the spatial dimension (shape: (bs,h*w*|srcs|,c))
        mask_flatten = torch.cat(mask_flatten, 1) # concat along the spatial dimension (shape: (bs,h*w*|srcs|))
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1) # concat along the spatial dimension (shape: (bs,h*w*|srcs|,c))


        # convert list to tensor
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        # level_start_index stores [0, w1*h1, w1*h1+w2*h2, ..., \sum{i=1}^{k-1}wi*ki] (Note: there are k feature levels) [currently k=1]
        level_start_index = torch.cat((spatial_shapes.new_zeros((1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
        # (used for accounting different sized imgs in a batch) ~mask stores loc info of original img [1 means true pixel, 0 means padded pixel] 
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1) # (Note: masks is a (bs,h,w) tensor)

        #ENCODER FUNC:
        def Encoder(src_flatten, spatial_shapes, level_start_index, valid_ratios, lvl_pos_embed_flatten, mask_flatten):
            #Encoder Layer:
            def Encoderlayer(src,pos,reference_points, spatial_shapes, level_start_index, padding_mask):
                '''
                src - (bs, h*w, c) [from backbone]
                pos - (bs, h*w, c)
                padding_mask - (bs, h*w) # for padding pixel identification
                '''
                query = src+pos
                src2 = self.def_attn(query, reference_points, src, spatial_shapes)
                src = src + self.dropout1(src2)
                src = self.norm1(src)
                # ffn
                src = self.forward_ffn(src)
                return src
            

            # For every feature map with features of shape (h,w), generate h*w points covering the entire map with distance between two pts being 0.5
            reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device)

            # iterate over multiple encoder layers (sequential - input of an encoder is the output of the previous encoder)
            encoder_output = src_flatten
            num_layers = 8 # number of encoder layers (different for different backbones)
            for _ in range(num_layers):
                # only output changes across encoder layers
                encoder_output = Encoderlayer(encoder_output, pos, reference_points, spatial_shapes, level_start_index, padding_mask)

            return encoder_output
        
        # encoder
        memory = Encoder(src_flatten, spatial_shapes, level_start_index, valid_ratios, lvl_pos_embed_flatten, mask_flatten)

        # DECODER FUNC:
        def Decoder(memory, mask_flatten):
            # Decoder Layer
            def DecoderLayer(tgt, query_pos, reference_points, src, src_spatial_shapes, src_padding_mask=None):
                ######### STANDARD TRANSFORMER OPERATIONS (self and cross attention) #########
                # self attention between decoder features (sort of)
                q = k = tgt + query_pos
                tgt2 = self.self_attn(q.transpose(0, 1), k.transpose(0, 1), tgt.transpose(0, 1))[0].transpose(0, 1) # normal attention
                tgt = tgt + self.dropout2(tgt2)
                tgt = self.norm2(tgt)

                # cross attention (Deformable) between encoder and decoder features
                tgt2 = self.cross_attn(tgt+query_pos, reference_points, src, src_spatial_shapes, src_padding_mask)
                tgt = tgt + self.dropout1(tgt2)
                tgt = self.norm1(tgt)

                # ffn
                tgt = self.forward_ffn(tgt)

                return tgt

            # get reference points (Decoder Style)
            bs, _, c = memory.shape
            num_queries = 100 # number of object queries or object proposals we desire to generate
            hidden_dim = 256 # number of channels in the feature map from the backbone
            query_pos = torch.nn.Embedding(num_queries, hidden_dim*2).weight # randomly initialized learnable weights of shape (bs, num_queries, 2*num_channels)

            query_pos, tgt = torch.split(query_pos, c, dim=1) # after split: query_embed - (bs, num_queries, num_channels), and tgt - (bs, num_queries, num_channels)
            reference_points = LinearTransformation(query_pos).sigmoid()
            init_reference_out = reference_points

            #
            num_layers = 8 # number of decoder layers
            output = tgt
            for _ in range(num_layers):
                reference_points_input = reference_points[:, :, None] * src_valid_ratios[:, None]
                output = DecoderLayer(output, query_pos, reference_points_input, memory, src_spatial_shapes, src_padding_mask)

            return output, reference_points, init_reference_out
        

        
        # decoder
        hs, inter_references, init_reference_out = Decoder(memory, mask_flatten)

        inter_references_out = inter_references
        return hs, init_reference_out, inter_references_out, None, None
    
        


        
        

def get_bboxes_and_logits(hs, init_reference, inter_references):
    outputs_classes = []
    outputs_coords = []
    for lvl in range(hs.shape[0]):
        if lvl == 0:
            reference = init_reference
        else:
            reference = inter_references[lvl - 1]
        reference = inverse_sigmoid(reference)
        outputs_class = self.class_embed[lvl](hs[lvl])
        tmp = self.bbox_embed[lvl](hs[lvl])
        if reference.shape[-1] == 4:
            tmp += reference
        else:
            assert reference.shape[-1] == 2
            tmp[..., :2] += reference
        outputs_coord = tmp.sigmoid()
        outputs_classes.append(outputs_class)
        outputs_coords.append(outputs_coord)
    outputs_class = torch.stack(outputs_classes)
    outputs_coord = torch.stack(outputs_coords)

    out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
    return out

def main(samples):
    
    '''
    backbone: output of a feature pyramid network where the pyramid is built of
              feature maps from last three stages of swin transformer 
              (page 4, Figure 3 in swin transformer paper) 
    '''
    features,pos = backbone(samples) # backbone -> SWINB (don't touch)
    srcs, masks, pos, query_embeds = compute_detr_inputs(features, samples, pos) # UNDERSTOOD
    hs, init_reference, inter_references, enc_outputs_class, enc_outputs_coord_unact = transformer(srcs, masks, pos, query_embeds)
    out = get_bboxes_and_logits(hs, init_reference, inter_references)
    return out
